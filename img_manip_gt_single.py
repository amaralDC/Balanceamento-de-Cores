import cv2 
import numpy as np

input_path = 'PLACEHOLDER'
input_image = 'PLACEHOLDER'
output_path = 'PLACEHOLDER'

# Ground truth reference

img = cv2.imread(input_path + input_image)

# Display coordinates on click.
def click_event(event, x, y, flags, params):
    # User provides coordinates by manually clicking the image.
    global h_start, w_start, h_width, w_width
    # Check for left mouse click.
    if event == cv2.EVENT_LBUTTONDOWN:
        # Display coordinates on console and window.
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        # Update coordinates
        h_start = y
        w_start = x
    # Check for right mouse click.
    if event==cv2.EVENT_RBUTTONDOWN:
        # Display coordinates on console and window.
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)
        # Update coordinates
        h_width = abs(y - h_start)
        w_width = abs(x - w_start)

img = cv2.imread(input_path + input_image, 1)
clone = img.copy() # Creating duplicate for annotations.

# Display clickable image.
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
# Setting mouse handler and calling click_event() function.
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image shape is h x w x 3
image = clone
image_patch = image[h_start:h_start+h_width, w_start:w_start+w_width]

# Get max pixel values from each channel (BGR) to normalize the original image. We assume the max pixel is white.
image_normalized = image / image_patch.max(axis=(0, 1))
print(image_normalized.max())
# Clip the values to be between 0 and 1
image_balanced = image_normalized.clip(0,1)

# Display the original and balanced images (optional)
# cv2.rectangle(clone, (w_start, h_start), (w_start+w_width, h_start+h_width), (0,0,255), 2)
# cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.imshow("Image", image)
# cv2.namedWindow('Image GT Balanced', cv2.WINDOW_NORMAL)
# cv2.imshow("Image GT Balanced", image_balanced)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save as 8 bit
image_balanced_8bit = (image_balanced*255).astype(int)
cv2.imwrite(output_path + "balanced_GT.png", image_balanced_8bit)