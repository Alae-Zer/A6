import os
from PIL import Image

def create_grayscale_samples(source_folder, target_folder):
    """
    Makes grayscale copies of the original samples.

    Args:
        source_folder: the folder containing the files to convert
        target_folder: the folder where the new versions are saved
    """
    # Make sure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Get all files from the source folder
    files = os.listdir(source_folder)
    counter = 0
    
    # Process each file
    for file_name in files:
        # Make sure we're only processing image files (assuming PNG format)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Get full path to the file
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)
            
            # Open the image and convert to grayscale
            image = Image.open(source_path)
            grayscale_image = image.convert("L")  # "L" mode is grayscale
            
            # Save the grayscale image
            grayscale_image.save(target_path)
            counter += 1
            
    print(f"Converted {counter} images to grayscale")


def build_bitmaps_and_targets(source_folder):
    """
    Creates bitmaps and targets.
    Each bitmap is a 32 x 32 binary matrix where 0 represents white and 1 represents black.
    Each target is a list of the digits that the corresponding samples represent.

    Args:
        source_folder: the folder containing the clean (black-and-white) samples

    Returns:
        bitmaps - N x 32 x 32 list
        targets - N x 1 list
    """
    # Lists to store our results
    bitmaps = []
    targets = []
    
    # Get all files from the source folder
    files = os.listdir(source_folder)
    
    # Process each file
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract the digit from the filename (format: digitX-sampleYYY.png)
            if file_name.startswith("digit") and "-" in file_name:
                digit = int(file_name[5])  # Character at index 5 is the digit
            else:
                print(f"Warning: Could not extract digit from filename: {file_name}")
                continue
            
            # Get full path to the file
            file_path = os.path.join(source_folder, file_name)
            
            # Open the grayscale image
            image = Image.open(file_path)
            
            # Convert to binary (0 for white, 1 for non-white)
            # Using a more tolerant threshold for white to reduce noise
            bitmap = []
            for y in range(32):
                row = []
                for x in range(32):
                    pixel = image.getpixel((x, y))
                    # Pixels with values >= 240 are considered white
                    if pixel >= 240:  
                        row.append(0)  # White pixel
                    else:
                        row.append(1)  # Non-white pixel
                bitmap.append(row)
            
            # Add to our lists
            bitmaps.append(bitmap)
            targets.append(digit)
    
    print(f"Processed {len(bitmaps)} images to bitmaps")
    return bitmaps, targets

def create_text_files(images, targets, target_folder):
    """
    Create text files where:
        The first line is the digit (0 - 9) that the image represents
        The next 32 lines are binary strings containing 32 binary digits

    The files should be renamed following the pattern

        digitX-sampleYYY.txt

    where X is the digit (0-9) represented, and YYY is the sample number,
    padded with 0s on the left if necessary to make three characters.

    Args:
        images: N x 32 x 32 list
        targets: N x 1 list
        targetFolder: the folder where the new text files are saved
    """
    # Make sure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Keep track of how many samples we've seen for each digit
    digit_counts = {}
    for i in range(10):
        digit_counts[i] = 0
    
    # Process each image
    for i in range(len(images)):
        # Get the target digit and increment its count
        digit = targets[i]
        digit_counts[digit] += 1
        
        # Create the file name with proper formatting
        sample_num = digit_counts[digit]
        file_name = f"digit{digit}-sample{sample_num:03d}.txt"
        file_path = os.path.join(target_folder, file_name)
        
        # Prepare the file content
        # First line is the digit
        content = [str(digit)]
        
        # Next 32 lines are the binary strings
        for row in images[i]:
            # Convert each row to a string of 0s and 1s
            binary_string = ''.join(str(bit) for bit in row)
            content.append(binary_string)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
    
    print(f"Created {len(images)} text files")


def analyze_image_values(source_folder):
    """Helper function to analyze pixel values in grayscale images"""
    # Get a sample image
    files = os.listdir(source_folder)
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_folder, file_name)
            image = Image.open(file_path)
            
            # Collect all pixel values
            min_val = 255
            max_val = 0
            for y in range(32):
                for x in range(32):
                    pixel = image.getpixel((x, y))
                    min_val = min(min_val, pixel)
                    max_val = max(max_val, pixel)
            
            print(f"Image: {file_name}")
            print(f"Min pixel value: {min_val}")
            print(f"Max pixel value: {max_val}")
            
            # Display histogram of values
            value_counts = {}
            for y in range(32):
                for x in range(32):
                    pixel = image.getpixel((x, y))
                    if pixel not in value_counts:
                        value_counts[pixel] = 0
                    value_counts[pixel] += 1
            
            print("Pixel value distribution:")
            for value in sorted(value_counts.keys()):
                print(f"  Value {value}: {value_counts[value]} pixels")
            
            # Only analyze one image for now
            break


### Main Driver ###
if __name__ == "__main__":
    original_folder = "originalSamples"
    grayscale_folder = "grayscaleSamples"
    text_folder = "textSamples"

    # Comment this out if you've already created grayscale samples
    create_grayscale_samples(original_folder, grayscale_folder)
    
    # Build bitmaps and targets
    bitmaps, targets = build_bitmaps_and_targets(grayscale_folder)
    
    # Create text files from bitmaps
    create_text_files(bitmaps, targets, text_folder)
    
    # Test specific example (digit2-sample039)
    sample_to_check = "digit2-sample039.png"
    if os.path.exists(os.path.join(grayscale_folder, sample_to_check)):
        image = Image.open(os.path.join(grayscale_folder, sample_to_check))
        
        # Apply our improved threshold
        bitmap = []
        for y in range(32):
            row = []
            for x in range(32):
                pixel = image.getpixel((x, y))
                if pixel >= 240:  
                    row.append(0)  # White pixel
                else:
                    row.append(1)  # Non-white pixel
            bitmap.append(row)
        
        print(f"\nTesting improved threshold on {sample_to_check}:")
        for row in bitmap:
            print(''.join(str(bit) for bit in row))
    
    print("Part 1 completed successfully!")