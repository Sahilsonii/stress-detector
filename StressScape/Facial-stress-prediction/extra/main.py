# import cv2
# import os
# import time
# import pandas as pd
# from datetime import datetime
# from deepface import DeepFace

# # Function to calculate stress %
# def get_stress_percentage(emotions):
#     stress_emotions = ['angry', 'disgust', 'fear', 'sad']
#     stress_score = sum([emotions.get(e, 0) for e in stress_emotions])
#     return round(stress_score, 2)

# # Ask for user name
# user_name = input("Enter user name: ").strip()
# base_dir = "stress_data"
# user_dir = os.path.join(base_dir, user_name)
# snapshot_dir = os.path.join(user_dir, "snapshots")

# os.makedirs(snapshot_dir, exist_ok=True)

# # CSV file path
# csv_path = os.path.join(user_dir, "stress_report.csv")

# # Create CSV file if not exist
# if not os.path.exists(csv_path):
#     df = pd.DataFrame(columns=["Timestamp", "Stress (%)"])
#     df.to_csv(csv_path, index=False)

# # Start webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ Error: Could not open webcam.")
#     exit()

# print("✅ Webcam started... Press 'q' to quit.")

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Camera read failed.")
#             break

#         # Capture timestamp
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#         # Save snapshot
#         image_path = os.path.join(snapshot_dir, f"{timestamp}.png")
#         cv2.imwrite(image_path, frame)

#         # Analyze emotions using DeepFace
#         try:
#             analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#             # DeepFace returns a dict (not list) in recent versions
#             if isinstance(analysis, list):
#                 analysis = analysis[0]
#             emotions = analysis['emotion']
#             stress_percent = get_stress_percentage(emotions)
#         except Exception as e:
#             print(f"⚠️ Face not detected or error: {str(e)}")
#             stress_percent = 0

#         # Save record in CSV
#         new_data = pd.DataFrame([[timestamp, stress_percent]], columns=["Timestamp", "Stress (%)"])
#         new_data.to_csv(csv_path, mode='a', header=False, index=False)

#         print(f"[{timestamp}] ✅ Stress: {stress_percent}% saved.")

#         # Wait for 60 seconds or exit on 'q'
#         print("⏳ Waiting for next snapshot... (press 'q' to stop)")
#         for i in range(60):
#             if cv2.waitKey(1000) & 0xFF == ord('q'):
#                 raise KeyboardInterrupt

# except KeyboardInterrupt:
#     print("\n🛑 Program stopped by user.")

# finally:
#     cap.release()
#     cv2.destroyAllWindows()

import cv2
import os
import time
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Function to calculate stress %
def get_stress_percentage(emotions):
    stress_emotions = ['angry', 'disgust', 'fear', 'sad']
    stress_score = sum(emotions.get(e, 0) for e in stress_emotions)
    return round(stress_score, 2)

# Ask for user name
user_name = input("Enter user name: ").strip()
if not user_name:
    print("❌ Error: User name cannot be empty.")
    exit()

base_dir = "stress_data"
user_dir = os.path.join(base_dir, user_name)
snapshot_dir = os.path.join(user_dir, "snapshots")
os.makedirs(snapshot_dir, exist_ok=True)

# CSV file path
csv_path = os.path.join(user_dir, "stress_report.csv")

# Create CSV file if not exists
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=["Timestamp", "Stress (%)"])
    df.to_csv(csv_path, index=False)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam started... Press 'q' to quit.")

# Set frame size for faster processing (optional but recommended)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera read failed.")
            break

        # Show frame (optional but helpful for user feedback)
        cv2.imshow('Stress Monitor - Press Q to Quit', frame)

        # Capture timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save snapshot
        image_path = os.path.join(snapshot_dir, f"{timestamp}.png")
        cv2.imwrite(image_path, frame)

        # Analyze emotions using DeepFace
        stress_percent = 0
        try:
            # DeepFace.analyze returns list of dicts; use silent=True to reduce logs
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            # Handle both single face (list[0]) and potential multiple faces
            if isinstance(result, list) and len(result) > 0:
                emotions = result[0]['emotion']
                stress_percent = get_stress_percentage(emotions)
            else:
                print("⚠️ No face detected.")
        except Exception as e:
            print(f"⚠️ Analysis error: {str(e)}")

        # Append to CSV efficiently
        with open(csv_path, 'a', newline='') as f:
            pd.DataFrame([[timestamp, stress_percent]], 
                         columns=["Timestamp", "Stress (%)"]).to_csv(f, header=False, index=False)

        print(f"[{timestamp}] ✅ Stress: {stress_percent}% saved.")

        # Wait 60 seconds with 1-second checks for 'q'
        print("⏳ Waiting 60 seconds for next snapshot... (press 'q' to stop)")
        start_time = time.time()
        while time.time() - start_time < 60:
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            time.sleep(0.1)  # Reduce CPU usage

except KeyboardInterrupt:
    print("\n🛑 Program stopped by user.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🧹 Resources cleaned up.")