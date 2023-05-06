import pytesseract
import cv2

# Path to the input image
image_path = 'images-115-max-keys-400/images/AT_Tablet/AT_Tablet0.jpg'
img = cv2.imread(image_path)
# Read the input image using OpenCV
def getText(img):
    

    # Preprocess the image to improve OCR accuracy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Perform OCR using pytesseract
    results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    print(text)
    # Loop over each recognized text region
    # for i in range(len(results['level'])):
    #     # Extract the bounding box coordinates for the text region
    #     x, y, w, h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
    #     # Draw the bounding box around the text region
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

getText(img)
# # Display the image with the bounding boxes
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()