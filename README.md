# SMS Spam Detection

Gelen SMS mesajlarını makine öğrenmesi ile **Spam** veya **Normal (Ham)** olarak sınıflandıran doğal dil işleme (NLP) projesi.

## Özet
* **Veri Seti:** UCI SMS Spam Collection (5.572 SMS).
* **Ön İşleme:** Gereksiz sütunlar ve etkisiz kelimeler temizlendi, metinler **TF-IDF** ile sayısallaştırıldı ve karakter uzunluğu (`message_length`) eklendi.
* **Model:** Multinomial Naive Bayes ve Logistic Regression karşılaştırıldı.

## Çalıştırma
```bash
pip install -r requirements.txt
python src/main.py
```
