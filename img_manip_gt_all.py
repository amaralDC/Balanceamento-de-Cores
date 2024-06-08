import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

input_path = 'C:/[...]/input/'
output_path = 'C:/[...]/output/GT/'
histogram_output_path = 'C:/[...]/output/GT/graph/'

# Ground truth reference

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
    image_gt_balanced = image_normalized.clip(0,1)

    # Save as uint 8 bit
    image_gt_balanced_8bit = (image_gt_balanced*255).astype(np.uint8)
    
    # Save the balanced image to output
    output_file_path = os.path.join(output_path, f"balanced_{file}")
    cv2.imwrite(output_file_path, image_gt_balanced_8bit)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagem original')
    axes[0, 0].axis('off')
    
    # White-balanced image
    axes[0, 1].imshow(cv2.cvtColor(image_gt_balanced_8bit, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Ground truth')
    axes[0, 1].axis('off')
    
    # Plot histograms for the original image
    plot_color_histograms(axes[1, 0], img, 'Imagem original')
    
    # Plot histograms for the white-balanced image
    plot_color_histograms(axes[1, 1], image_gt_balanced_8bit, 'Ground truth')
    
    plt.tight_layout()
    
    i += 1
    # Save the histogram
    histogram_file_path = os.path.join(histogram_output_path, f"histogram_GT_{i}.png")
    plt.savefig(histogram_file_path)
    # Close the figure to avoid display
    plt.close(fig)