import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    # Načítanie dát
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Predspracovanie textu (tokenizácia a vektorizácia)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['question'])
    sequences = tokenizer.texts_to_sequences(df['question'])
    X = pad_sequences(sequences, padding='post')

    y = pd.get_dummies(df['query_id']).values
    return X, y, tokenizer
