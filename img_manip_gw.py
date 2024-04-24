import os
import cv2
import numpy as np

input_path = 'PLACEHOLDER'
output_path = 'PLACEHOLDER'

# Gray world assumption

def GW_white_balance(img):
    # Convert the image to LAB color space.
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Calculate the mean color values in A and B channels.
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    # Subtract 128 (mid gray) from the averages and normalize the L channel by multiplying with the difference.
    # Then, subtract this value from A and B channels. 
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2) # Add a multiplication factor to increase/decrease
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2) # the overall brightness (here, * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

for file in os.listdir(input_path):
    # print(f"Processing image: {file}")
    input_file_path = os.path.join(input_path, file)
    # print(f"Image path: {input_file_path}")
    img = cv2.imread(input_file_path)
    
    image_gw_balanced = GW_white_balance(img)
    
    # Display the original and balanced images (optional)
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", img)
    # cv2.namedWindow('GW Balanced', cv2.WINDOW_NORMAL)
    # cv2.imshow("GW Balanced", image_gw_balanced)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save the balanced image to output
    output_file_path = os.path.join(output_path, f"balanced_{file}")
    # print(f"Saving balanced image to: {output_file_path}")
    cv2.imwrite(output_file_path, image_gw_balanced)
    # print("Image saved successfully.")