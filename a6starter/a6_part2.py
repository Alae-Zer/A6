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
    pass


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
    pass


def flatten_images(images):
    """
    Convert each two-dimensional matrix to a one-dimensional list.

    Args:
        bitmaps - N x S x S list of two-dimensional matrices

    Returns:
        N x (S^2) list of one-dimensional lists of length S*S
    """
    pass


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


### Main Driver ###
if __name__ == "__main__":
    source_folder = "textSamples"
    reports_folder = "reports"
    testing_percentage = 0.25

    images, targets = read_images_and_targets_from_text_files(source_folder)
    images = reduce_dimensions(images)
    images = flatten_images(images)
    train_predict_report(images, targets, testing_percentage, reports_folder)
