import os
import pandas as pd
from PIL import Image
import cv2

def save_data(image1, base_path, second, vid_type):
    folder_path = os.path.join(base_path, 'Comparing')
    
    if not os.path.exists(folder_path):
        # Create the Comparing folder if it doesn't exist
        os.makedirs(folder_path)

    files_second_path = os.path.join(folder_path, str(second))
    
    if not os.path.exists(files_second_path):
        # Create the second folder if it doesn't exist
        os.makedirs(files_second_path)

    # Convert images to RGB format
    try:
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error occurred while converting images to RGB: {e}")
        return

    # Save images to output folder
    if vid_type == 1:
        vid_type='Etalon.png'
    else:
        vid_type='User_uploaded.png'
    try:
        img1 = Image.fromarray(img1)
        img1_path = os.path.join(files_second_path, vid_type)
        
        # Overwrite the existing images
        if os.path.exists(img1_path):
            os.remove(img1_path)
        
        img1.save(img1_path)
        print(f"Image saved to: {files_second_path}")
    except Exception as e:
        print(f"Error occurred while saving image: {e}")
    return
