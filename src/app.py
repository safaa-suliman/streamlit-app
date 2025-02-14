import re
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.tokenize import sent_tokenize
from utils.text_processing import preprocess_text
import nltk
nltk.download('punkt')



def extract_dates(text):
    date_pattern = (r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                   r'[-/,.\s]*\d{1,2}[-/,\s]*\d{2,4}\b|\b\d{1,2}[-/,\s]*\d{1,2}[-/,\s]*\d{2,4}\b')
    return re.findall(date_pattern, text, re.IGNORECASE)

def analyze_texts(pdf_texts, top_n, language='english'):
    all_text = " ".join(doc["text"] for doc in pdf_texts)
    filtered_words = preprocess_text(all_text, language)
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n), word_counts

def analyze_texts_by_date(pdf_texts, top_n, language='english', period='yearly'):
    date_word_counts = {}
    for doc in pdf_texts:
        text = doc["text"]
        dates = extract_dates(text)
        filtered_words = preprocess_text(text, language)
        word_counts = Counter(filtered_words)
        
        for date in dates:
            date_obj = None
            for fmt in ('%d %b %Y', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d'):
                try:
                    date_obj = datetime.strptime(date, fmt)
                    break
                except ValueError:
                    continue
            
            if not date_obj:
                continue
            
            if period == 'yearly':
                date_key = date_obj.year
            elif period == 'quarterly':
                date_key = f"{date_obj.year}-Q{(date_obj.month - 1) // 3 + 1}"
            elif period == 'half-yearly':
                date_key = f"{date_obj.year}-H{(date_obj.month - 1) // 6 + 1}"
            elif period == '3-years':
                date_key = f"{date_obj.year // 3 * 3}-{date_obj.year // 3 * 3 + 2}"
            elif period == '5-years':
                date_key = f"{date_obj.year // 5 * 5}-{date_obj.year // 5 * 5 + 4}"
            else:
                date_key = date_obj.year
            
            if date_key not in date_word_counts:
                date_word_counts[date_key] = Counter()
            date_word_counts[date_key].update(word_counts)
    
    return {date: counts.most_common(top_n) for date, counts in date_word_counts.items()}

def nmf_topic_modeling_with_sentences(pdf_texts, num_topics=3):
    texts = [doc["text"] for doc in pdf_texts]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    dtm = vectorizer.fit_transform(texts)
    
    nmf = NMF(n_components=num_topics, random_state=42, init='nndsvd')
    nmf.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        sentences = []
        
        for text in texts:
            for sentence in sent_tokenize(text):
                if any(word in sentence for word in topic_words):
                    sentences.append(sentence)
                    if len(sentences) >= 2:
                        break
            if len(sentences) >= 2:
                break
        
        topics.append(f"Topic {topic_idx + 1}: {' '.join(sentences)}")
    
    return topics
