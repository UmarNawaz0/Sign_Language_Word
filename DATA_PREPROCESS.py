# preprocess_landmark_data.py
import os
import pickle
import numpy as np

DATA_DIR = './data_landmarks'
OUTPUT_FILE = 'landmark_data.pickle'
SEQUENCE_LENGTH = 30

sequences = []
labels = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.pkl'):
        file_path = os.path.join(DATA_DIR, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if len(data['landmarks']) == SEQUENCE_LENGTH:
                sequences.append(data['landmarks'])
                labels.append(data['label'])
            else:
                print(f"Skipping {file}: {len(data['landmarks'])} frames (expected {SEQUENCE_LENGTH})")

sequences = np.array(sequences)
labels = np.array(labels)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': sequences, 'labels': labels}, f)
print(f"Dataset created with {len(sequences)} sequences.")

# Verify
with open(OUTPUT_FILE, 'rb') as f:
    data = pickle.load(f)
    print("Files:", os.listdir(DATA_DIR))
    print("Labels:", data['labels'])