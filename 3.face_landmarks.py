import dlib
import cv2

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
image = cv2.imread('musk.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Load the landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract and save each detected face
for i, face in enumerate(faces):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # Crop the face from the original image
    cropped_face = image[y:y+h, x:x+w]
    
    # Save the cropped face as a new jpg file
    face_filename = f"face_{i+1}.jpg"
    cv2.imwrite(face_filename, cropped_face)
    print(f"Face saved as {face_filename}")

    # Optionally, you can also draw rectangles and landmarks for visualization
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Detect landmarks for each face
    landmarks = predictor(gray, face)

    # Draw circles for each landmark
    for n in range(0, 68):  # 68 is the number of landmarks
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

# Display the image with landmarks and rectangles (optional)
cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
