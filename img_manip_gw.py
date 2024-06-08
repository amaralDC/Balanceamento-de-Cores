import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = 'C:/[...]/input/'
output_path = 'C:/[...]/output/GW/'
histogram_output_path = 'C:/[...]/output/GW/graph/'

# Gray world assumption

def GW_white_balance(img):
    # Convert the image to LAB color space.
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Calculate the mean color values in A and B channels.
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    # Subtract 128 (mid gray) from the averages and normalize the L channel by multiplying with the difference.
    # Then, subtract this value from A and B channels. 
    # Optionally, add a multiplication factor to increase/decrease overall brightness. Here, * 1.2.
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

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
    # print(f"Processing image: {file}")
    input_file_path = os.path.join(input_path, file)
    # print(f"Image path: {input_file_path}")
    img = cv2.imread(input_file_path)
    
    image_gw_balanced = GW_white_balance(img)

    # Save the balanced image to output
    output_file_path = os.path.join(output_path, f"balanced_{file}")
    cv2.imwrite(output_file_path, image_gw_balanced)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagem original')
    axes[0, 0].axis('off')
    
    # White-balanced image
    axes[0, 1].imshow(cv2.cvtColor(image_gw_balanced, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Gray world')
    axes[0, 1].axis('off')
    
    # Plot histograms for the original image
    plot_color_histograms(axes[1, 0], img, 'Imagem original')
    
    # Plot histograms for the white-balanced image
    plot_color_histograms(axes[1, 1], image_gw_balanced, 'Gray world')
    
    plt.tight_layout()
    
    i += 1
    # Save the histogram
    histogram_file_path = os.path.join(histogram_output_path, f"histogram_GW_checker_{i}.png")
    plt.savefig(histogram_file_path)
    # Close the figure to avoid display
    plt.close(fig)