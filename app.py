import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON
with open("emotiondetectionsrt.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotiondetectionsrt.weights.h5")

# Load the Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)  # Make sure to use the correct camera index

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a named window
cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop to capture frames and perform emotion detection
while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Resize the face image to match model input size
        face_resized = cv2.resize(face, (48, 48))

        # Extract features and normalize
        face_features = extract_features(face_resized)

        # Predict emotion
        pred = model.predict(face_features)
        prediction_label = labels[pred.argmax()]

        # Display predicted emotion label
        cv2.putText(frame, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the output frame in full screen
    cv2.imshow("Output", frame)

    # Check for ESC key press to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()