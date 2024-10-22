import cv2
import pickle
import numpy as np
import os

# Create data folder if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
faces_data = []
i = 0
name = input("Enter your Aadhar number: ")
frames_total = 51
capture_after_frame = 2

def save_faces_data(name, faces_data):
    """Save face data and name into pickle files."""
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape((frames_total, -1))

    # Save name into names.pkl
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * frames_total
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * frames_total
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save face data into faces_data.pkl
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

def process_frame(frame, faces_data):
    """Process each frame for face detection and data collection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= frames_total and i % capture_after_frame == 0:
            faces_data.append(resized_img)
        cv2.putText(frame, f'Captures: {len(faces_data)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    return frame

# Main loop for video capture and face data collection
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Process frame for face detection and capture
    frame = process_frame(frame, faces_data)

    # Save the output frame (remove if unnecessary)
    cv2.imwrite('output_frame.jpg', frame)

    # Display the frame
    cv2.imshow('Video Frame', frame)

    # Exit on 'q' key or when enough frames are captured
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= frames_total:
        break

# Release video and destroy windows
video.release()
cv2.destroyAllWindows()

# Save the collected face data and Aadhar number
save_faces_data(name, faces_data)

print(f"Data captured for {name} and saved successfully.")
