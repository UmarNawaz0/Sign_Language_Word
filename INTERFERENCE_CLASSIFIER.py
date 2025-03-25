# inference_landmark.py
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

MODEL_FILE = 'landmark_model.h5'
SEQUENCE_LENGTH = 30
ANALYSIS_DURATION = 5
FPS_ESTIMATE = 30
OUTPUT_DISPLAY_DURATION = 5
TOTAL_FRAMES = ANALYSIS_DURATION * FPS_ESTIMATE

word_map = {0: "Other", 1: "Around the House"}
VALID_LABELS = [1]
MIN_CONFIDENCE = 0.75  # Slightly higher to reject false positives

model = load_model(MODEL_FILE)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
frame_buffer = []
recognized_word = "None"
output_display_time = None
analysis_start_time = None
analyzing = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

print("Press 'S' to start 5-second gesture analysis. Only 'Around the House' is valid. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            padding = 20
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w) - padding)
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h) - padding)
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w) + padding)
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h) + padding)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not analyzing:
        analyzing = True
        analysis_start_time = time.time()
        frame_buffer = []
        out = cv2.VideoWriter('test_gesture.avi', fourcc, FPS_ESTIMATE, (w, h))
        print("Starting 5-second gesture analysis...")

    if analyzing:
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)
            frame_buffer.append(landmarks[:126])
        else:
            frame_buffer.append([0] * 126)

        out.write(frame)

        elapsed_time = time.time() - analysis_start_time
        if elapsed_time >= ANALYSIS_DURATION:
            if len(frame_buffer) >= SEQUENCE_LENGTH:
                step = len(frame_buffer) // SEQUENCE_LENGTH
                sampled_frames = frame_buffer[::step][:SEQUENCE_LENGTH]
                input_sequence = np.expand_dims(sampled_frames, axis=0)
                prediction = model.predict(input_sequence, verbose=0)[0]
                print("Raw prediction scores:", prediction)
                predicted_label = np.argmax(prediction)

                if predicted_label in VALID_LABELS and prediction[predicted_label] >= MIN_CONFIDENCE:
                    recognized_word = word_map[predicted_label]
                    print(f"Predicted label: {predicted_label}, Recognized word: {recognized_word}")
                    engine.say(recognized_word)
                    engine.runAndWait()
                else:
                    recognized_word = "Invalid Gesture"
                    print(
                        f"Predicted label: {predicted_label}, Recognized word: {recognized_word} - Not 'Around the House' or low confidence")

                output_display_time = time.time()

            analyzing = False
            out.release()
            out = None
            print("Analysis complete.")

    if analyzing:
        remaining_time = max(0, int(ANALYSIS_DURATION - (time.time() - analysis_start_time)))
        cv2.putText(frame, f"Analyzing: {remaining_time}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Press 'S' to analyze", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if output_display_time and (time.time() - output_display_time) <= OUTPUT_DISPLAY_DURATION:
        color = (0, 255, 0) if recognized_word != "Invalid Gesture" else (0, 0, 255)
        cv2.putText(frame, f"Output: {recognized_word}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        recognized_word = "None"
        output_display_time = None

    cv2.imshow("PSL Gesture Recognition", frame)

    if key == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()