import sys
from src.preprocessing import load_data, preprocess_data
from src.model import build_model
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # Načítanie dát
    file_path = "data/dataset.csv"
    df = load_data(file_path)

    # Predspracovanie dát
    X, y, tokenizer = preprocess_data(df)

    # Definovanie modelu
    model = build_model(input_dim=len(tokenizer.word_index) + 1, max_length=X.shape[1], output_dim=y.shape[1])

    # Tréning modelu
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2, callbacks=[early_stopping])

    # Uloženie modelu
    model.save("model.h5")
    print("Model saved as model.h5")

if __name__ == "__main__":
    main()
