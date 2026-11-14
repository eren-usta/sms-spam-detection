import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data

def evaluate_models():
    print("Veri yükleniyor ve modeller değerlendiriliyor...")
    df = load_data('data/raw/sms_spam.csv')
    
    X = df['message']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Her iki modeli de test etmek için tanımlıyoruz
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n{'='*10} {name} Değerlendirmesi {'='*10}")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))

    # --- Feature Importance (Özellik Önem Düzeyi) Grafiği ---
    print("\nFeature Importance (Özellik Önem Düzeyi) grafiği oluşturuluyor...")
    log_reg_model = LogisticRegression(random_state=42)
    log_reg_model.fit(X_train_vec, y_train)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = log_reg_model.coef_[0]
    
    importance_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    top_spam_features = importance_df.sort_values(by='coefficient', ascending=False).head(15)
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_spam_features['feature'], top_spam_features['coefficient'], color='crimson')
    plt.xlabel('Katsayı Değeri (Feature Importance)')
    plt.title('Spam Tahmininde En Etkili Kelimeler')
    plt.gca().invert_yaxis()  # En yüksek olan en üstte dursun
    
    output_path = 'reports/figures/feature_importance.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Özellik önem düzeyi grafiği '{output_path}' olarak kaydedildi!")

if __name__ == "__main__":
    evaluate_models()