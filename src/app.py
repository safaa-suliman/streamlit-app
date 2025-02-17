import streamlit as st
import re
import nltk
import PyPDF2
import plotly.express as px
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')
import nltk
nltk.download('punkt')


# Function to preprocess text
def preprocess_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.isalnum() and word not in stop_words]

# Function to extract dates
def extract_dates(text):
    date_pattern = (r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                   r'[-/,."]*\d{1,2}[-/,"]*\d{2,4}\b|\b\d{1,2}[-/,"]*\d{1,2}[-/,"]*\d{2,4}\b')
    return re.findall(date_pattern, text, re.IGNORECASE)

# Function for topic modeling
def nmf_topic_modeling(texts, num_topics=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    dtm = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=num_topics, random_state=42, init='nndsvd')
    nmf.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(topic_words)}")
    return topics

# Function for sentiment analysis
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Streamlit UI
st.title("🔍 Advanced Text Analysis Tool")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
if uploaded_file:
    file_text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            file_text += page.extract_text()
    else:
        file_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("📜 Extracted Text Preview")
    st.write(file_text[:1000] + "...")
    
    # Preprocess & Analyze
    processed_text = preprocess_text(file_text)
    word_counts = Counter(processed_text)
    most_common_words = word_counts.most_common(10)
    
    # Display word frequency
    st.subheader("📊 Word Frequency Analysis")
    fig = px.bar(x=[w[0] for w in most_common_words], y=[w[1] for w in most_common_words], title="Top 10 Most Frequent Words")
    st.plotly_chart(fig)
    
    # Extract Dates
    extracted_dates = extract_dates(file_text)
    st.subheader("📅 Extracted Dates")
    st.write(extracted_dates if extracted_dates else "No dates found.")
    
    # Topic Modeling
    topics = nmf_topic_modeling([file_text])
    st.subheader("📌 Topic Modeling")
    for topic in topics:
        st.write(topic)
    
    # Sentiment Analysis
    sentiment = sentiment_analysis(file_text)
    st.subheader("😊 Sentiment Analysis")
    st.write(f"Overall Sentiment: {sentiment}")
