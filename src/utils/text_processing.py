import os
import re
import nltk
from nltk_data.corpus import stopwords
from nltk_data.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'nltk_data')
nltk.data.path.append(nltk_data_path)
required_nltk_resources = ['punkt', 'stopwords']

for resource in required_nltk_resources:
    try:
        nltk.data.find(f'{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

def preprocess_text(text, language='english'):
    stop_words = set(stopwords.words(language))
    linking_words = set(['and', 'or', 'but', 'so', 'because', 'however', 'therefore', 'moreover', 'thus', 'hence'])
    words = word_tokenize(re.sub(r'\W+', ' ', text.lower()))  # Tokenize and clean text
    return [word for word in words if word.isalnum() and word not in stop_words and word not in linking_words]
