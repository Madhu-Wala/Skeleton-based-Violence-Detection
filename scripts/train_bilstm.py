import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.dataset_loader import load_dataset

# PARAMETERS
MAX_FRAMES = 50
BATCH_SIZE = 32
EPOCHS = 40
FEATURES = 165  # full motion

#  Load data
X_train, y_train = load_dataset("extracted_keypoints/train",max_frames=MAX_FRAMES,add_motion=True)
X_val, y_val = load_dataset("extracted_keypoints/val",max_frames=MAX_FRAMES,add_motion=True)

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# BUILD MODEL 
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(MAX_FRAMES, FEATURES)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# TRAIN 
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# SAVE MODEL 
model.save("bilstm_model_165.h5")
print("✅ Training complete. Model saved as bilstm_model_165.h5")
