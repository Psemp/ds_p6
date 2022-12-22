import PIL.Image
import numpy as np

from PIL import ImageOps, ImageFilter


def additionnal_cleaning(product_image: PIL.Image, mode: str = "gs_he") -> PIL.Image:
    """
    Function : Takes an image and a mode into argument (default : gs_he) and
    applies autoconstrast, then applies the transformation specified as mode :
    - gs = grayscale
    - gs_he = grayscale, equalization of histograms
    - rgb = returns image as is
    - rgb_he = equalization of histograms

    Args :
    - product_image : PIL.Image, the image to transform
    - mode : str, the mode of transformation

    Returns :
    product_image : PIL.Image with transformations applied
    """

    mode = mode.lower()
    product_image = ImageOps.autocontrast(image=product_image)

    match mode:
        case "gs":
            product_image = ImageOps.grayscale(image=product_image)
        case "gs_he":
            product_image = ImageOps.grayscale(image=product_image)
            product_image = ImageOps.equalize(image=product_image)
        case "rgb":
            pass
        case "rgb_he":
            product_image = ImageOps.equalize(image=product_image)
        case _:
            raise Exception(f"No mode prodived or mode : '{mode}' not recognized, cf. docstring")

    return product_image


def noise_and_blur(product_image, mode: str = "L"):
    """
    Add noise and blur to an image.

    Args:
    product_image PIL.Image, The image to be noise and blur.

    Returns:
    filtered_image : PIL.Image, The modified image.

    """
    product_image_arr = np.array(product_image)
    noise = np.random.normal(0, 5, product_image_arr.shape)
    noisy_img = PIL.Image.fromarray(product_image_arr + noise).convert(mode)

    filtered_image = noisy_img.filter(ImageFilter.BoxBlur(0.5))
    return filtered_image
