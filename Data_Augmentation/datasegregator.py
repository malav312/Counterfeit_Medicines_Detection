import os
import shutil

# Path to the main image directory
image_dir = 'classifier_dataset/images'

# Loop through all medicine subdirectories
for medicine_dir in os.listdir(image_dir):
    # Ignore non-directory files
    print("Medicine Dir",medicine_dir)
    if not os.path.isdir(os.path.join(image_dir, medicine_dir)):
        continue

    # Create real and fake subdirectories within the medicine subdirectory
    real_dir = os.path.join(image_dir, medicine_dir, 'real')
    fake_dir = os.path.join(image_dir, medicine_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Loop through all images in the medicine subdirectory
    for image_file in os.listdir(os.path.join(image_dir, medicine_dir)):
        # Ignore non-image files
        if not image_file.endswith('.jpg'):
            continue

        # Determine if the image is real or fake based on the file name
        if '_blur' in image_file or '_gaussian_noise' in image_file or '_rotated' in image_file:
            # Image is fake, move to fake subdirectory
            shutil.move(os.path.join(image_dir, medicine_dir, image_file), os.path.join(fake_dir, image_file))
            print("Moved to Fake",image_file)
        else:
            # Image is real, move to real subdirectory
            shutil.move(os.path.join(image_dir, medicine_dir, image_file), os.path.join(real_dir, image_file))
            print("Moved to Real",image_file)