import cv2
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Set the path for the dataset
dataset_path = 'dataset'

# Create the directory if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH recognizer (using the standard method)
recognizer = cv2.face_LBPHFaceRecognizer_create()

# Function to collect data for training the model
def collect_data():
    face_id = input("Enter your ID for dataset collection: ")
    print("Collecting data for face ID:", face_id)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{dataset_path}/User.{face_id}.{count}.jpg", face)

        cv2.imshow("Collecting Data", frame)

        if count >= 50:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Data collection for ID {face_id} completed!")

# Function to train the model
def train_model():
    faces = []
    labels = []
    label_dict = {}

    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]

    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_id = int(image_path.split('.')[1])  # Extract face ID from filename
        label_dict[face_id] = image_path.split('.')[1]

        faces.append(gray)
        labels.append(face_id)

    # Convert to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)

    # Train the LBPH model
    recognizer.train(faces, labels)

    # Save the model for later use
    recognizer.save("face_trained_model.yml")

    print("Training completed!")

# Function to recognize face and mark attendance
def mark_attendance(id):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{id},{dt_string}\n")

# Function to test and recognize faces using the trained model
def recognize_face():
    recognizer.read("face_trained_model.yml")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(face)

            # Mark attendance if recognized
            if label_id != -1:
                print(f"Recognized face ID: {label_id}")
                mark_attendance(label_id)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Recognizing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("1. Collect Data")
        print("2. Train Model")
        print("3. Recognize Face and Mark Attendance")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_face()
        elif choice == '4':
            break
        else:
            print("Invalid choice! Please select again.")
