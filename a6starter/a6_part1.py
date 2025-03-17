import os
from PIL import Image

def create_grayscale_samples(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    files = os.listdir(source_folder)
    counter = 0
    
    for file_name in files:
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
            
        image = Image.open(source_path)
        grayscale_image = image.convert("L")  # "L" mode is grayscale
            
        grayscale_image.save(target_path)
        counter += 1
            
    print(f"Converted {counter} images to grayscale")


def build_bitmaps_and_targets(source_folder):
    bitmaps = []
    targets = []
    
    files = os.listdir(source_folder)
    
    for file_name in files:
            if file_name.startswith("digit") and "-" in file_name:
                digit = int(file_name[5])
            else:
                print(f"Warning: Could not extract digit from filename: {file_name}")
                continue
            
            file_path = os.path.join(source_folder, file_name)
            
            # Open the grayscale image
            image = Image.open(file_path)

            bitmap = []
            for y in range(32):
                row = []
                for x in range(32):
                    pixel = image.getpixel((x, y))
                    if pixel >= 240:  
                        row.append(0)
                    else:
                        row.append(1)
                bitmap.append(row)
            
            bitmaps.append(bitmap)
            targets.append(digit)
    
    print(f"Processed {len(bitmaps)} images to bitmaps")
    return bitmaps, targets

def create_text_files(images, targets, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    digit_counts = {}
    for i in range(10):
        digit_counts[i] = 0
    
    for i in range(len(images)):
        digit = targets[i]
        digit_counts[digit] += 1
        
        sample_num = digit_counts[digit]
        file_name = f"digit{digit}-sample{sample_num:03d}.txt"
        file_path = os.path.join(target_folder, file_name)
        
        content = [str(digit)]
        
        for row in images[i]:
            binary_string = ''.join(str(bit) for bit in row)
            content.append(binary_string)
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
    
    print(f"Created {len(images)} text files")



### Main Driver ###
if __name__ == "__main__":
    original_folder = "originalSamples"
    grayscale_folder = "grayscaleSamples"
    text_folder = "textSamples"

    create_grayscale_samples(original_folder, grayscale_folder)

    bitmaps, targets = build_bitmaps_and_targets(grayscale_folder)

    create_text_files(bitmaps, targets, text_folder)
    
    print("Part 1 completed successfully!")