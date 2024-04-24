import os
import cv2 
import numpy as np

input_path = 'PLACEHOLDER'
output_path = 'PLACEHOLDER'

# White patch algorithm
        
# Find color checker, then determine white patch.
def find_patch(image):
    # BUG Not processing the gray stone pictures (white patches are blurry)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    white_patch = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Adjust according to image. Orig = 300
            x, y, w, h = cv2.boundingRect(contour)
            if w > 15 and h > 15:  # Adjust according to image. Orig = 15  15
                white_patch = (x, y, w, h)
                break
    return white_patch

for file in os.listdir(input_path):
    input_file_path = os.path.join(input_path, file)
    img = cv2.imread(input_file_path, 1)
    clone = img.copy() # Creating duplicate for annotations.

    patch = find_patch(clone)
    if patch is not None:
        w_start, h_start, w_width, h_width = patch

        # Image shape is h x w x 3
        image = clone
        image_patch = image[h_start:h_start+h_width, w_start:w_start+w_width]

        # Get max pixel values from each channel (BGR) to normalize the original image. We assume the max pixel is white.
        image_normalized = image / image_patch.max(axis=(0, 1))
        print(image_normalized.max())
        # Clip the values to be between 0 and 1
        image_wp_balanced = image_normalized.clip(0,1)

        # Save as 8 bit
        image_wp_balanced_8bit = (image_wp_balanced*255).astype(int)
        
        # Save the balanced image to output
        output_file_path = os.path.join(output_path, f"balanced_{file}")
        cv2.imwrite(output_file_path, image_wp_balanced_8bit)