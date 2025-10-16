import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")

# Folder containing PNG images
folder_path = "/src/apifootball_model/comps_val_acc"

# Get a list of PNG files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Sort image files (optional, based on alphabetical order)
image_files.sort()

# Define the number of rows
rows_per_column = 4

# Calculate the number of columns needed
num_columns = (len(image_files) + rows_per_column - 1) // rows_per_column

# Create a figure for the plot
fig, axes = plt.subplots(rows_per_column, num_columns, figsize=(num_columns * 6, rows_per_column * 6))

# Iterate over the images and plot them column by column
for col in range(num_columns):
    for row in range(rows_per_column):
        img_index = col * rows_per_column + row
        if img_index < len(image_files):
            img_path = os.path.join(folder_path, image_files[img_index])
            img = mpimg.imread(img_path)

            # Display the image in the corresponding subplot
            axes[row, col].imshow(img)
            axes[row, col].axis("off")  # Hide the axis
        else:
            # Hide unused subplots
            axes[row, col].axis("off")

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(folder_path, "result.png"))
