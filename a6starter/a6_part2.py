import os
from FileUtils import readIntoList

def read_images_and_targets_from_text_files(source_folder):
    """
    Reads images and targets from data stored in text files.
    It is expected that each text file has 33 lines:
        the first line is the target digit
        each of the remaining 32 lines is a 32-bit binary string

    Args:
        source_folder: the folder containing the text files

    Returns:
        images - N x 32 x 32 list of image data
        targets - N x 1 list of targets
    """
    images = []
    targets = []
    
    # Get all files from the source folder
    files = os.listdir(source_folder)
    
    # Process each file
    for file_name in files:
        if file_name.lower().endswith('.txt'):
            file_path = os.path.join(source_folder, file_name)
            
            try:
                # Read the file using the utility function
                lines = readIntoList(file_path)
                
                # First line is the target digit
                target = int(lines[0])
                
                # Next 32 lines are the binary bitmap
                bitmap = []
                for i in range(1, 33):  # Skip the first line (target)
                    # Convert each character to an integer
                    row = [int(bit) for bit in lines[i]]
                    bitmap.append(row)
                
                # Add to our lists
                images.append(bitmap)
                targets.append(target)
                
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    print(f"Read {len(images)} images from text files")
    return images, targets


def reduce_dimensions(bitmaps):
    """
    Create new matrices of the indicated size, where each number in the matrix
    is the number of 1's each in non-overlapping 4x4 block in the original bitmap.

    Args:
        bitmaps - N x 32 x 32 list of image data

    Returns:
        N x 8 x 8 list of bit counts
    """
    image_size = 32
    reduced_size = 8
    block_size = image_size // reduced_size  # 4x4 blocks
    
    reduced_bitmaps = []
    
    for bitmap in bitmaps:
        reduced_bitmap = []
        
        for i in range(reduced_size):  # 8 rows in reduced bitmap
            row = []
            for j in range(reduced_size):  # 8 columns in reduced bitmap
                # Count 1's in the current 4x4 block
                count = 0
                for bi in range(block_size):
                    for bj in range(block_size):
                        # Calculate position in original bitmap
                        orig_i = i * block_size + bi
                        orig_j = j * block_size + bj
                        count += bitmap[orig_i][orig_j]
                
                row.append(count)
            reduced_bitmap.append(row)
            
        reduced_bitmaps.append(reduced_bitmap)
    
    print(f"Reduced dimensions of {len(bitmaps)} bitmaps from 32x32 to 8x8")
    return reduced_bitmaps


def flatten_images(images):
    """
    Convert each two-dimensional matrix to a one-dimensional list.

    Args:
        bitmaps - N x S x S list of two-dimensional matrices

    Returns:
        N x (S^2) list of one-dimensional lists of length S*S
    """
    flattened_images = []
    
    for image in images:
        # Flatten the 2D matrix into a 1D list
        flat_image = []
        for row in image:
            flat_image.extend(row)
        
        flattened_images.append(flat_image)
    
    if len(images) > 0:
        original_size = f"{len(images[0])}x{len(images[0][0])}"
        flat_size = len(flattened_images[0])
        print(f"Flattened {len(images)} images from {original_size} to {flat_size}")
    
    return flattened_images


def train_predict_report(images, targets, test_percentage, reports_folder):
    """
    Train four different classifiers (algorithms) and produce reports for each.
        1. Decision Tree
        2. Gaussian Naive Bayes
        3. K Nearest Neighbors
        4. Support Vector Machine

    Args:
        images (list of size N x 1 x 64): reduced-dimension, flattened bitmaps
        targets (list of size N x 1): the target (correct classification) of each image
        test_percentage (float): what percentage of the data set to use as testing data
        reports_folder (string): where to put the PNG files containing the reports
    """
    pass


if __name__ == "__main__":
    source_folder = "textSamples"
    reports_folder = "reports"
    testing_percentage = 0.25

    # Read the bitmap text files
    images, targets = read_images_and_targets_from_text_files(source_folder)
    
    # Reduce dimensions from 32x32 to 8x8
    reduced_images = reduce_dimensions(images)
    
    # Check result of dimension reduction
    if len(reduced_images) > 0:
        print(f"Dimensions after reduction: {len(reduced_images[0])} x {len(reduced_images[0][0])}")
        print("Sample of first reduced image (8x8):")
        for row in reduced_images[247]:
            print(' '.join(f"{val:2d}" for val in row))
    
    # Flatten images from 8x8 to 1x64
    flattened_images = flatten_images(reduced_images)
    
    # Check result of flattening
    if len(flattened_images) > 0:
        print(f"Length of flattened image: {len(flattened_images[0])}")
        print(f"First 10 values of flattened image: {flattened_images[0][:10]}")
    
    # The train_predict_report function will be implemented next
    # train_predict_report(flattened_images, targets, testing_percentage, reports_folder)
        