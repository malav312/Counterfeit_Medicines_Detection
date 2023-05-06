import cv2
import pytesseract

def get_text_coordinates(image_path):
    # Load image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to preprocess the image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Run pytesseract on the preprocessed image
    data = pytesseract.image_to_data(
        thresh, output_type=pytesseract.Output.DICT)

    # Get the bounding boxes around the text
    boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            (x, y, w, h) = (data['left'][i], data['top']
                            [i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h))

            # Draw bounding box around the text
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the image with the bounding box
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes

get_text_coordinates('../input/meds/images/A2__Tablet/A2__Tablet2.jpg')
