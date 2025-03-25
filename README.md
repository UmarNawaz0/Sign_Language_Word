# Sign_Language_Word
This project is a two-handed sign language interpreter designed to recognize the phrase "Around the House" using machine learning and deep learning techniques. It utilizes MediaPipe for hand landmark detection and a LSTM (Long Short-Term Memory) neural network model for sequence classification.

**Installation**

**Prerequisites**

Make sure you have Python 3.8 or later installed along with the following libraries:

**pip install opencv-python numpy tensorflow mediapipe pyttsx3 scikit-learn**
Make sure these libraries are install

**Usage**

**1. Data Collection**

To collect landmark data for "Around the House" and other gestures:

python DATA_COLLECTION.py

First 10 recordings: Exact PSL gesture for "Around the House".

Next 10 recordings: Other PSL gestures and random two-handed movements.

**2. Data Preprocessing**

Process the recorded data into a structured dataset:

python DATA_PREPROCESS.py

Creates landmark_data.pickle for training.

**3. Model Training**

Train the LSTM model for gesture recognition:

python MODEL_TRAIN.py

Trains the model and saves it as landmark_model.h5.

**4. Real-Time Inference**

Perform real-time gesture recognition:

python INTERFERENCE_CLASSIFIER.py

Press 'S' to start 5-second gesture analysis.

Recognized gestures will be vocalized.

**How to Add New Words**

Modify DATA_COLLECTION.py:

Update VIDEO_COUNT and gesture_names to include new words.

**Record Data:**

Collect 20 videos per new word (10 exact gestures, 10 other movements).

Run Data Preprocessing:

Process data to update landmark_data.pickle.

**Retrain the Model:**

Rerun MODEL_TRAIN.py for new classifications.

I have uploaded a dataset of landmarks, just use that but it is only for single word, so make sure to expand this more, add few more words
