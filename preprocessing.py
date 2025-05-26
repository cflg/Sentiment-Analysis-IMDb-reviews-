import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clear_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

def apply_preprocessing(df):
    df['text'] = df['text'].apply(clear_text)
    return df
