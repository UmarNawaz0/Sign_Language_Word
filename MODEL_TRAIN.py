# train_landmark_model.py
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

SEQUENCE_LENGTH = 30
LANDMARK_SIZE = 126

with open('landmark_data.pickle', 'rb') as f:
    data = pickle.load(f)

sequences = np.array(data['data'])
labels = np.array(data['labels'])

num_classes = 2
labels = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(128, input_shape=(SEQUENCE_LENGTH, LANDMARK_SIZE), return_sequences=True),
    LSTM(128),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=4, epochs=40, validation_data=(X_test, y_test))
model.save('landmark_model.h5')
print("Model training complete and saved.")