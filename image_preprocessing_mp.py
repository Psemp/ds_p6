import os
import glob
import concurrent.futures
import warnings

from PIL import Image, ImageOps, ImageFilter

image_dir = "./data/Images"

image_paths = glob.glob(pathname=f"{image_dir}/*")


def preprocessing(image_path):
    """
    Preprocesses the image at the given path by transforming it into a square image and resizing it to 128x128.
    Applies a Gaussian filter to remove potential noise. Adjust the filter via radius parameter (line 55)

    Args:
    - image_path: path to image for pillow to process

    If the image is not already a square, creates a new image, fills it with black and pastes the
    original image centered. The resulting image is then resized to 128x128. Uses a Gaussian Filter to
    improve image quality.

    Returns:
    Nothing
    Saves the preprocessed image in the "./imgs/" (+ mode) directory.
    """
    size = 124  # Image size, always a square
    mode = "gs_he"
    outdir = f"./imgs/{mode}"
    image_name = os.path.basename(image_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = Image.open(image_path)

    width, height = image.size

    image = ImageOps.autocontrast(image=image)
    image = ImageOps.equalize(image=image)
    if width == height:
        pass
    elif width > height:
        new_image = Image.new(image.mode, (width, width), (0, 0, 0))
        new_image.paste(image, (0, (width - height) // 2))
        image = new_image
    else:
        new_image = Image.new(image.mode, (height, height), (0, 0, 0))
        new_image.paste(image, ((height - width) // 2, 0))
        image = new_image

    image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    if mode == "gs" or mode == "gs_he":
        image = image.convert(mode="L")

    gaussian_filter = ImageFilter.GaussianBlur(radius=0.35)
    image = image.filter(filter=gaussian_filter)

    image.save(fp=f"{outdir}/{image_name}")


workers = os.cpu_count()


if __name__ == "__main__":
    print("Running ...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(preprocessing, image_paths)

    print("Done")
