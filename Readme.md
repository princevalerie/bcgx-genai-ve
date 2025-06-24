
---

# Financial PDF Analyzer 📊

A Streamlit-based web application that leverages Google Gemini AI and LangChain to analyze financial PDF reports. Upload your financial statements and interactively ask questions about their contents with advanced natural language capabilities.

## Features

- **PDF Upload**: Upload multiple financial PDF reports for processing.
- **AI-Powered Analysis**: Uses Google's Gemini model for answering financial queries.
- **Document Search**: Vector-based retrieval ensures answers are grounded in your uploaded PDFs.
- **Chat Interface**: Conversational interface for asking follow-up questions.
- **Example Questions**: Get started quickly with built-in financial queries.


## Getting Started

### Prerequisites

- Python 3.9+
- Google Cloud API key with access to Gemini models

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/princevalerie/bcgx-genai-ve.git
   cd bcgx-genai-ve
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google API Key:**
   - Obtain your API key from Google Cloud Console.
   - You will enter this in the Streamlit sidebar when running the app.

### Running the App

```bash
streamlit run app.py
```

Open the displayed local URL in your browser.

## Usage

1. **Enter your Google API Key** in the sidebar.
2. **Upload one or more PDF files** (e.g., financial reports).
3. Wait for processing to complete.
4. Ask questions about your uploaded data in the chat interface, such as:
   - "What is the revenue trend over the last 3 years?"
   - "What are the main financial risks identified?"

## File Structure

- `app.py` - Main Streamlit application.
- `requirements.txt` - Python dependencies.

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [PyMuPDF4LLM](https://pypi.org/project/langchain-pymupdf4llm/) for PDF parsing

## Notes

- Your API key is required at runtime; it is not stored.
- Only PDF files are supported.
- The app is intended for educational or research use; review outputs carefully for critical decisions.
