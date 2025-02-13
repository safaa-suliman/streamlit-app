from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def clustering(pdf_texts, num_clusters=3):
    if len(pdf_texts) < num_clusters:
        raise ValueError("Number of clusters cannot be greater than the number of documents.")
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in pdf_texts])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(tfidf_matrix)
    return kmeans.labels_
