import streamlit as st
import PyPDF2
import nltk
from textblob import TextBlob

# Ensure necessary resources are available
nltk.download('punkt')

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def analyze_sentiment(text):
    """Perform basic sentiment analysis."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Streamlit UI
st.title("📂 Multi-File Upload and Analysis")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")

        # Extract text
        file_text = extract_text_from_pdf(file)
        if file_text:
            st.text_area("Extracted Text", file_text[:1000] + "...", height=200)

            # Analyze sentiment
            polarity, subjectivity = analyze_sentiment(file_text)
            st.write(f"**Sentiment Analysis:**")
            st.write(f"🔹 **Polarity:** {polarity} (Positive/Negative)")
            st.write(f"🔹 **Subjectivity:** {subjectivity} (Objective/Subjective)")
