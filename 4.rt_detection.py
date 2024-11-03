import cv2
import dlib

# Function to extract and draw rectangles around the eyes
def extract_eyes(landmarks, frame):
    # Left eye coordinates (landmark points 36 to 41)
    lx_min = min(landmarks.part(i).x for i in range(36, 42))
    ly_min = min(landmarks.part(i).y for i in range(36, 42))
    lx_max = max(landmarks.part(i).x for i in range(36, 42))
    ly_max = max(landmarks.part(i).y for i in range(36, 42))
    cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), (0, 255, 255), 2)

    # Right eye coordinates (landmark points 42 to 47)
    rx_min = min(landmarks.part(i).x for i in range(42, 48))
    ry_min = min(landmarks.part(i).y for i in range(42, 48))
    rx_max = max(landmarks.part(i).x for i in range(42, 48))
    ry_max = max(landmarks.part(i).y for i in range(42, 48))
    cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), (0, 255, 255), 2)

# Function to extract and draw a rectangle around the mouth
def extract_mouth(landmarks, frame):
    # Mouth coordinates (landmark points 48 to 67)
    mx_min = min(landmarks.part(i).x for i in range(48, 68))
    my_min = min(landmarks.part(i).y for i in range(48, 68))
    mx_max = max(landmarks.part(i).x for i in range(48, 68))
    my_max = max(landmarks.part(i).y for i in range(48, 68))
    cv2.rectangle(frame, (mx_min, my_min), (mx_max, my_max), (255, 0, 255), 2)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start webcam feed
cap = cv2.VideoCapture(0) # this number may be 0, 1, 2, ... depending on your system config.

while True:
    ret, frame = cap.read()  # Capture frame-by-frame from the webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Draw bounding box around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Extract and draw rectangles around the eyes and mouth (Step 7)
        extract_eyes(landmarks, frame)
        extract_mouth(landmarks, frame)

    # Display the frame with detection and landmarks
    cv2.imshow('Webcam Face Detection and Landmarking', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()