import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_vectorize(df):
    """
    Mesaj metinlerini TF-IDF kullanarak sayısal matrise dönüştürür.
    """
    # X (özellikler) ve y (etiketler) olarak ayıralım
    X = df['message']
    y = df['label']
    
    # TF-IDF Vectorizer uygulayalım (keşifte kullandığımız parametrelerle)
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)
    
    return X_tfidf, y, tfidf

if __name__ == "__main__":
    from data_loader import load_data
    
    # Test edelim
    df = load_data('data/raw/sms_spam.csv')
    X_tfidf, y, tfidf = preprocess_and_vectorize(df)
    print("Ön işleme tamamlandı. Matris boyutu:", X_tfidf.shape)