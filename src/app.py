import streamlit as st
import PyPDF2
import nltk
from collections import Counter
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure necessary resources are available
nltk.download('punkt')
nltk.download('stopwords')



def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {e}")
        return None

def preprocess_text(text):
    """Tokenize and clean text."""
    words = nltk.word_tokenize(text.lower())  # Convert to lowercase
    words = [word for word in words if word.isalnum()]  # Remove punctuation
    return words

def analyze_word_frequency(words, top_n=10):
    """Return top N word frequencies."""
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

def generate_wordcloud(words):
    """Generate a word cloud from words."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    return wordcloud

def analyze_sentiment(text):
    """Perform sentiment analysis."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Streamlit UI
st.title("📂 Multi-File Upload and Word Frequency Analysis")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_words = []
    document_word_counts = {}

    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")

        # Extract text
        file_text = extract_text_from_pdf(file)
        if file_text:
            st.text_area("Extracted Text", file_text[:1000] + "...", height=150)

            # Process text
            words = preprocess_text(file_text)
            all_words.extend(words)
            document_word_counts[file.name] = analyze_word_frequency(words, top_n=10)

            # Display Top Words for the document
            st.write(f"**Top Words in {file.name}:**")
            for word, count in document_word_counts[file.name]:
                st.write(f"🔹 {word}: {count} times")

            # Sentiment Analysis
            polarity, subjectivity = analyze_sentiment(file_text)
            st.write(f"**Sentiment Analysis:**")
            st.write(f"🔹 **Polarity:** {polarity} (Positive/Negative)")
            st.write(f"🔹 **Subjectivity:** {subjectivity} (Objective/Subjective)")

            # Generate and show Word Cloud
            st.write("**Word Cloud:**")
            fig, ax = plt.subplots()
            ax.imshow(generate_wordcloud(words), interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # **Overall Analysis**
    if all_words:
        st.subheader("📊 **Overall Top Words Across All Documents**")
        overall_top_words = analyze_word_frequency(all_words, top_n=10)
        for word, count in overall_top_words:
            st.write(f"🔹 {word}: {count} times")

        # Overall Word Cloud
        st.write("**Overall Word Cloud:**")
        fig, ax = plt.subplots()
        ax.imshow(generate_wordcloud(all_words), interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
