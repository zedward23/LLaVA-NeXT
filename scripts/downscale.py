import os
from PIL import Image

def downscale_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image.save(image_path)
    return image_path


#for all elements in a folder, create a new folder with the same name and downscale all images in the folder
def downscale_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            os.makedirs(os.path.join(folder_path, dir + "_downscaled"))
        for file in files:
            downscale_image(os.path.join(folder_path, file))
    return folder_path

downscale_folder("images/images")