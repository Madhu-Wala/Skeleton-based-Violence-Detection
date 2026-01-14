from keras.models import Sequential
from keras.layers import (
    Input, Masking, Bidirectional, LSTM, Dense, Dropout
)
from keras.optimizers import Adam

def build_bilstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Masking(mask_value=0.0),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),

        Bidirectional(LSTM(64)),
        Dropout(0.4),

        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
