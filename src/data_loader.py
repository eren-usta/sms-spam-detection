import pandas as pd

def load_data(file_path):
    """
    Verilen yoldaki CSV dosyasını latin1 encoding ile yükler ve ilk iki sütunu seçer.
    """
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    return df

if __name__== "__main__":
    # Test amaçlı çalıştırma (dosya yolu ana dizine göredir)
    df = load_data('data/raw/sms_spam.csv')
    print("Veri başarıyla yüklendi. Boyut:", df.shape)