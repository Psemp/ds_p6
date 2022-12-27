import os
import glob
import concurrent.futures
import skimage
import warnings

image_dir = "./imgs/gs_he"

image_paths = glob.glob(pathname=f"{image_dir}/*")


def noise_and_blur(image_path):
    image_array = skimage.io.imread(fname=image_path)
    image_array = skimage.util.random_noise(image_array, var=0.0025)
    image_array = skimage.filters.gaussian(image_array, sigma=0.5)
    with warnings.catch_warnings():  # Doesnt suppress the warning but it works
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname=image_path, arr=image_array)


workers = os.cpu_count()

# for image_path in image_paths:  # debug
#     noise_and_blur(image_path=image_path)


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(noise_and_blur, image_paths)
