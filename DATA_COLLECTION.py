# record_landmark_data.py
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle

DATA_DIR = './data_landmarks'
os.makedirs(DATA_DIR, exist_ok=True)

VIDEO_DURATION = 5
FPS = 30
SEQUENCE_LENGTH = 30
VIDEO_COUNT = 20  # 10 "Around the House", 10 "Other" (including PSL gestures)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
video_index = 0

print("Press 'S' to record. First 10: 'Around the House' (exact PSL gesture). Next 10: 'Other' (Courtyard, Swimming Pool, Channel, Basement, random two-handed). Press 'ESC' to quit.")
while video_index < VIDEO_COUNT:
    ret, frame = cap.read()
    if not ret:
        print("Camera error. Exiting...")
        break
    cv2.imshow("Press 'S' to Record", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if video_index < 10:
            filename = f"around_house_{video_index + 1}.pkl"
            label = 1  # "Around the House"
        else:
            # Include PSL gestures and random two-handed movements in "Other"
            gesture_names = [
                'courtyard', 'swimming_pool', 'channel', 'basement', 'random_1',
                'twist_both', 'flap_both', 'cross_both', 'wave_both', 'clap_both'
            ]
            filename = f"other_{gesture_names[video_index - 10]}_{video_index - 9}.pkl"
            label = 0  # "Other"

        file_path = os.path.join(DATA_DIR, filename)

        print(f"Recording {filename}...")
        sequence = []
        frame_count = 0
        total_frames = VIDEO_DURATION * FPS
        step = total_frames // SEQUENCE_LENGTH

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                if len(results.multi_hand_landmarks) == 1:
                    landmarks.extend([0] * 63)
                if frame_count % step == 0 and len(sequence) < SEQUENCE_LENGTH:
                    sequence.append(landmarks[:126])
            else:
                if frame_count % step == 0 and len(sequence) < SEQUENCE_LENGTH:
                    sequence.append([0] * 126)
            frame_count += 1
            cv2.imshow("Recording...", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if len(sequence) == SEQUENCE_LENGTH:
            with open(file_path, 'wb') as f:
                pickle.dump({'landmarks': sequence, 'label': label}, f)
            print(f"Saved {filename} with {len(sequence)} frames.")
            video_index += 1
        else:
            print(f"Failed to record {filename}: only {len(sequence)} frames captured.")

    if key == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Data collection complete.")