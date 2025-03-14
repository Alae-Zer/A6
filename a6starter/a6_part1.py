def create_grayscale_samples(source_folder, target_folder):
    """
    Makes grayscale copies of the original samples.

    Args:
        source_folder: the folder containing the files to convert
        target_folder: the folder where the new versions are saved
    """
    pass


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
    pass


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
    pass


### Main Driver ###
if __name__ == "__main__":
    original_folder = "originalSamples"
    grayscale_folder = "grayscaleSamples"
    text_folder = "textSamples"

    create_grayscale_samples(original_folder, grayscale_folder)
    bitmaps, targets = build_bitmaps_and_targets(grayscale_folder)
    create_text_files(bitmaps, targets, text_folder)
