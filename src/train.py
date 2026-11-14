import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# src klasöründeki modülleri sorunsuz bulabilmek için
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

def train_and_evaluate_models():
    print("Veri yükleniyor ve hazırlanıyor...")
    df = load_data('data/raw/sms_spam.csv')
    
    # Olası boş mesaj satırlarını temizleme
    df = df.dropna(subset=['message', 'label'])
    
    # Özellik ve etiket ayırma
    X = df['message']
    y = df['label']
    
    # Eğitim ve test setine bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vektörleştirme
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Modelleri tanımlama
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\n--- Model Karşılaştırma ve Çapraz Doğrulama ---")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1_weighted')
        print(f"{name} - 5 Katlı Çapraz Doğrulama F1 Ortalaması: {cv_scores.mean():.4f}")
        
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        score = f1_score(y_test, y_pred, pos_label='spam')
        print(f"{name} - Test F1-Score (Spam): {score:.4f}")
        print("-" * 40)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\nEn iyi performans gösteren model: {best_name} (F1-Score: {best_score:.4f})")
    
    # Modeli ve vektörleştiriciyi kaydetme
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("En iyi model ve vectorizer 'models/' klasörüne kaydedildi!")

if __name__ == "__main__":
    train_and_evaluate_models()
