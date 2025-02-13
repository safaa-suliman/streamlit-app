import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'nltk_data')
nltk.data.path.append(nltk_data_path)

required_nltk_resources = ['punkt', 'stopwords']

for resource in required_nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
    except LookupError:
        try:
            nltk.download(resource, download_dir=nltk_data_path)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

def preprocess_text(text, language='english'):
    """Tokenizes, removes stopwords, and cleans the input text."""
    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        print(f"Stopwords for language '{language}' not found. Using English as default.")
        stop_words = set(stopwords.words('english'))

    words = word_tokenize(text.lower(), preserve_line=True)  # Better tokenization
    cleaned_words = [word for word in words if word.isalnum() and word not in stop_words]

    return cleaned_words
