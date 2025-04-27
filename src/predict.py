import sys
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_data

def predict(query):
    model = load_model("model.h5")

    # Predspracovanie dotazu
    df = pd.DataFrame({"question": [query]})
    X, _, tokenizer = preprocess_data(df)

    # Predikcia
    predicted = model.predict(X)
    predicted_query_id = pd.DataFrame(predicted).idxmax(axis=1)[0]

    return predicted_query_id

if __name__ == "__main__":
    query = sys.argv[1]  # Dotaz bude zadaný ako argument pri spustení
    result = predict(query)
    print(f"Predpovedaný SPARQL dotaz: {result}")
