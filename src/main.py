import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train_and_evaluate_models

def predict_sms(message):
    """
    Girilen örnek bir SMS metninin spam olup olmadığını eğitilmiş model ile tahmin eder.
    """
    model_path = 'models/best_model.pkl'
    vec_path = 'models/vectorizer.pkl'
    
    # Eğer modeller henüz eğitilmediyse otomatik eğitimi başlat
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("Eğitilmiş model bulunamadı, önce model eğitimi başlatılıyor...\n")
        train_and_evaluate_models()
        print("\n" + "="*40 + "\n")
        
    model = joblib.load(model_path)
    tfidf = joblib.load(vec_path)
    
    # Metni vektöre dönüştürelim
    message_tfidf = tfidf.transform([message])
    
    # Tahmin yapalım
    prediction = model.predict(message_tfidf)[0]
    return prediction

if __name__ == "__main__":
    test_messages = [
        "Congratulations! You've won a free ticket to Bahamas. Call now!",
        "Hey, are we still meeting for lunch today?"
    ]
    
    print("--- SMS Spam Tahmin Testi ---")
    for msg in test_messages:
        result = predict_sms(msg)
        print(f"Mesaj: '{msg}'\nSonuç: {result}\n")
