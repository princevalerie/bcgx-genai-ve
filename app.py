import os
import tempfile
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import google.generativeai as genai
from PIL import Image
import io
import base64

# Streamlit Configuration
st.set_page_config(
    page_title="Financial PDF Analyzer",
    page_icon="📊",
    layout="wide"
)

# SIDEBAR CONFIGURATION
st.sidebar.title("⚙️ Configuration")

# Input API Key
api_key = st.sidebar.text_input("Enter Google API Key:", type="password")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your Google API Key to continue.")
    # Main content area - only show title when no API key
    st.title("📊 AI Assistant for Financial Analysis")
    st.info("👈 Please enter your Google API Key in the sidebar to get started.")
    st.stop()

# Gemini Configuration
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Initialize models without caching to avoid issues
@st.cache_resource
def initialize_models():
    try:
        # Model for text and multimodal
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        
        # Multimodal model for images
        multimodal_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Embeddings for vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        return model, multimodal_model, embeddings
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing models: {str(e)}")
        return None, None, None

model, multimodal_model, embeddings = initialize_models()

if not model:
    st.title("📊 AI Assistant for Financial Analysis")
    st.error("Failed to initialize AI models. Please check your API key.")
    st.stop()

# SIDEBAR - FILE UPLOAD SECTION
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload Documents")

# Upload file in sidebar
uploaded_files = st.sidebar.file_uploader(
    "Upload Financial Report PDF Files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF financial report files for analysis"
)

# SIDEBAR - INSTRUCTIONS
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 How to Use")
st.sidebar.markdown("""
1. ✅ Enter your Google API Key above
2. 📄 Upload PDF financial reports
3. ⏳ Wait for processing to complete
4. 💬 Ask questions about the data
""")

# Financial analysis prompt template
prompt_template = """You are a financial analyst. Analyze the documents and answer the question concisely.
            
            Documents: {context}
            
            Question: {question}
            
            Provide clear financial analysis with specific data from the documents. If information is missing, state limitations clearly."""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def process_pdf(uploaded_file):
    """Process uploaded PDF file using PyMuPDF4LLM"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Use PyMuPDF4LLM loader
            loader = PyMuPDF4LLMLoader(tmp_file_path)
            documents = loader.load()
            
            # Extract text content from documents
            text_content = ""
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text_content += doc.page_content + "\n"
                elif hasattr(doc, 'text'):
                    text_content += doc.text + "\n"
                else:
                    text_content += str(doc) + "\n"
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            # Return text content or fallback message
            if text_content and text_content.strip():
                return text_content
            else:
                return "No readable content found in PDF. The document may be empty or corrupted."
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ PyMuPDF4LLM conversion failed for {uploaded_file.name}, trying fallback method")
            
            # Fallback to pypdf if PyMuPDF4LLM fails
            try:
                import pypdf
                with open(tmp_file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    fallback_text = ""
                    for page in pdf_reader.pages:
                        fallback_text += page.extract_text() + "\n"
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    if fallback_text.strip():
                        return fallback_text
                    else:
                        return "No readable text found in PDF. The document may contain only images or be password protected."
            except Exception as fallback_e:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                return "Failed to extract content from PDF using both primary and fallback methods."
        
    except Exception as e:
        st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        return None

def split_text(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def create_vector_store(texts):
    """Create vector store from texts"""
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    except Exception as e:
        st.sidebar.error(f"❌ Error creating vector store: {str(e)}")
        return None

def retrieve_docs(vector_store, query, k=5):
    """Retrieve relevant documents"""
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        st.sidebar.error(f"❌ Error retrieving documents: {str(e)}")
        return []

def answer_question(question, documents):
    """Answer questions based on documents"""
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Create chain with Gemini model
        prompt_formatted = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt_formatted | model
        
        response = chain.invoke({"question": question, "context": context})
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        st.sidebar.error(f"❌ Error answering question: {str(e)}")
        return "Sorry, an error occurred while processing your question."

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# SIDEBAR - PROCESSING SECTION
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Processing Status")

# Process uploaded files
if uploaded_files and not st.session_state.documents_processed:
    # Reset processing state
    st.session_state.documents_processed = False
    
    # Show processing status in sidebar
    with st.sidebar:
        st.info("🔄 Processing documents...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_texts = []
        processing_success = True
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name}")
            
            try:
                text = process_pdf(uploaded_file)
                
                if text and text.strip():
                    chunked_texts = split_text(text)
                    all_texts.extend(chunked_texts)
                    st.success(f"✅ {uploaded_file.name} ({len(chunked_texts)} chunks)")
                else:
                    st.warning(f"⚠️ No text from {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed: {uploaded_file.name}")
                processing_success = False
                break
        
        if all_texts and processing_success:
            # Create vector store
            status_text.text("Creating searchable index...")
            vector_store = create_vector_store(all_texts)
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.documents_processed = True
                st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
                st.balloons()
                progress_bar.empty()
                status_text.empty()
            else:
                st.error("❌ Failed to create searchable index")
        else:
            st.error("❌ Processing failed")

# SIDEBAR - STATUS INDICATOR
if st.session_state.documents_processed:
    st.sidebar.success("✅ Documents ready for analysis")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset & Upload New Files", type="secondary"):
        st.session_state.documents_processed = False
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.rerun()
    
    # SIDEBAR - EXAMPLE QUESTIONS
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Example Questions")
    example_questions = [
        "What is the revenue trend over the last 3 years?",
        "What is the current liquidity ratio?",
        "What are the main financial risks identified?",
        "How is the company's profitability performance?",
        "Provide investment recommendations based on this report"
    ]
    
    for i, eq in enumerate(example_questions):
        if st.sidebar.button(eq, key=f"example_{i}", type="secondary"):
            # Add to chat history and process
            st.session_state.chat_history.append({"role": "user", "content": eq})
            try:
                related_documents = retrieve_docs(st.session_state.vector_store, eq)
                if related_documents:
                    answer = answer_question(eq, related_documents)
                else:
                    answer = "I couldn't find relevant information in the documents to answer your question."
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.rerun()

elif uploaded_files:
    st.sidebar.info("⏳ Processing in progress...")
else:
    st.sidebar.info("📄 No documents uploaded yet")

# MAIN CONTENT AREA - CLEAN LAYOUT
st.title("📊 AI Assistant for Financial Analysis")

# Only show chat interface if documents are processed
if st.session_state.documents_processed and st.session_state.vector_store:
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
    
    # Input question
    if question := st.chat_input("Ask about financial data in your PDFs..."):
        # Add user question to history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
        
        # Process and answer question
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    related_documents = retrieve_docs(st.session_state.vector_store, question)
                    if related_documents:
                        answer = answer_question(question, related_documents)
                    else:
                        answer = "I couldn't find relevant information in the documents to answer your question."
                    
                    st.write(answer)
                    
                    # Add answer to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Sorry, an error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

else:
    # Show welcome message when no documents are processed
    if not uploaded_files:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h3>Welcome to Financial PDF Analyzer</h3>
            <p>👈 Upload your financial PDF reports in the sidebar to get started</p>
            <p>Once processed, you can ask questions about:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>📈 Revenue and profit trends</li>
                <li>💰 Financial ratios and metrics</li>
                <li>📊 Balance sheet analysis</li>
                <li>⚠️ Risk assessments</li>
                <li>💡 Investment recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.documents_processed:
        st.info("⏳ Please wait while your documents are being processed...")
