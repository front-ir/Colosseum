import cv2
import os

script_path = os.path.dirname(os.path.realpath(__file__))

def load_classifier(cascade_path):
    # Load the Haar cascade file
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError("Cannot load cascade classifier from file: " + cascade_path)
    return cascade

def detect_and_display(cascade, image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("Cannot open image file: " + image_path)

    # Convert to grayscale for detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform detection
    detections = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    # Draw rectangles around detected objects
    for (x, y, w, h) in detections:
        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print (x + (w * 0.5), y + (h * 0.5))
        #cv2.rectangle(grey_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #print(f"Detected at: x={x + w * 0.5}, y={y + h * 0.5}")

    # Display the result
    cv2.imshow('Detected Objects', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the classifier with provided path
    cascade = load_classifier(f'{script_path}haar-white-shahed/cascade.xml')

    # Path to the image to detect on
    image_path = 'U:\projects\Blocks_5.2.1_VS\screenshots\\1.png'

    # Perform detection and display results
    detect_and_display(cascade, image_path)
