import easyocr


def get_text(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    text = ' '.join([word[1] for word in result])
    return text


text = get_text('dolo.jpg')
print('Text:', text)
