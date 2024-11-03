import dlib
import cv2


# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
image = cv2.imread('musk.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Draw rectangles around detected faces
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the output
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()