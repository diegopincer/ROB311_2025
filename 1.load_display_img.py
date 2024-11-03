import cv2

# Load a sample image
image = cv2.imread('musk.jpg')

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()