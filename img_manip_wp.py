import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

input_path = 'C:/[...]/input/'
output_path = 'C:/[...]/output/WP/'
histogram_output_path = 'C:/[...]/output/WP/graph/'

# White patch algorithm
        
# Find color checker, then determine white patch.
def find_patch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adjust the lower and upper thresholds here. The lower it is, the darker the patch used.
    # 255 is max as it is white. We choose 160 ~ 180 for min through analysis of the marble photos.
    _, thresh = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    white_patch = None
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjust according to image. Orig = 300
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust according to image. Orig = 15  15
            if w > 5 and h > 5:
                white_patch = (x, y, w, h)
                break
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     x, y, w, h = cv2.boundingRect(contour)
    #     white_patch = (x, y, w, h)
    #     break
    return white_patch

def plot_color_histograms(ax, img, title):
    colors = ('b', 'g', 'r')
    ax.set_title(f'Histograma - {title}')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Píxeis')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
        ax.set_xlim([0, 256])

i = 0
for file in os.listdir(input_path):
    input_file_path = os.path.join(input_path, file)
    img = cv2.imread(input_file_path, 1)
    # Creating duplicate for annotations.
    clone = img.copy()

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

        # Save as uint 8 bit
        image_wp_balanced_8bit = (image_wp_balanced*255).astype(np.uint8)
        
        # Save the balanced image to output
        output_file_path = os.path.join(output_path, f"balanced_{file}")
        cv2.imwrite(output_file_path, image_wp_balanced_8bit)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Imagem original')
        axes[0, 0].axis('off')
        
        # White-balanced image
        axes[0, 1].imshow(cv2.cvtColor(image_wp_balanced_8bit, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('White patch')
        axes[0, 1].axis('off')
        
        # Plot histograms for the original image
        plot_color_histograms(axes[1, 0], img, 'Imagem original')
        
        # Plot histograms for the white-balanced image
        plot_color_histograms(axes[1, 1], image_wp_balanced_8bit, 'White patch')
        
        plt.tight_layout()
        
        i += 1
        # Save the histogram
        histogram_file_path = os.path.join(histogram_output_path, f"histogram_WP_checker_{i}.png")
        plt.savefig(histogram_file_path)
        # Close the figure to avoid display
        plt.close(fig)