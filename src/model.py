from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_model(input_dim, max_length, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=50, input_length=max_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))  # Počet tried = počet rôznych predpripravených dotazov
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
