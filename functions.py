from PIL import Image, ImageOps, ImageFilter
import cv2, os
import numpy as np

##############################################################################################################################

PATH_WATERMARK = 'watermark.png'
PATH_FOLDER_RAW = 'images_raw'
PATH_FOLDER_EDITED = 'images_edited'

##############################################################################################################################

def get_folder_images(folder):

    # raw folder directory
    path_folder_raw = f'{PATH_FOLDER_RAW}/{folder}'

    # create folder to store edited images
    path_folder_edited = f'{PATH_FOLDER_EDITED}/{folder}'
    if not os.path.exists(path_folder_edited):
        os.makedirs(path_folder_edited)

    # find all images in raw folder
    # filter to only image type files
    list_images = [image for image in os.listdir(path_folder_raw) if (('.jpeg' in image) or ('.jpg' in image) or ('.png' in image))]

    return list_images, path_folder_raw, path_folder_edited

##############################################################################################################################

def resize_image(image, target, by=''):

    w, h = image.size

    if by=='height':
        scale = target / h
        target_w, target_h = int(w * scale), target
    elif by=='width':
        scale = target / w
        target_w, target_h = target, int(h * scale)
    
    image_resized = image.resize((target_w, target_h), Image.ANTIALIAS)

    return image_resized

##############################################################################################################################

def crop_image(image, target_w, target_h):
    
    w, h = image.size
    crop_size_w = int((w - target_w)/2)
    crop_size_h = int((h - target_h)/2)
    
    image_cropped = ImageOps.crop(image, (crop_size_w, crop_size_h, crop_size_w, crop_size_h))

    return image_cropped

##############################################################################################################################

def crop_image_square(image, length_sides, by=''):
    
    w, h = image.size

    # left, top, right, bottom
    if by=='height':
        crop_size = int((w - length_sides)/2)
        border = (crop_size, 0, crop_size, 0)

    elif by=='width':
        crop_size = int((h - length_sides)/2)
        border = (0, crop_size, 0, crop_size)
    
    image_square = ImageOps.crop(image, border)

    return image_square

##############################################################################################################################

def combine_images(image_back, image_front):

    w_back, h_back = image_back.size
    w_front, h_front = image_front.size

    image_combined = image_back.filter(ImageFilter.GaussianBlur(15))
    image_combined.paste(image_front, (int((w_back - w_front)/2), int((h_back - h_front)/2)))
    
    return image_combined

##############################################################################################################################

def watermark_image(image_combined, scale_watermark=None):

    # read watermark image as cv2 format
    image_watermark = cv2.imread(PATH_WATERMARK)

    if scale_watermark!=None:

        # re-scale watermark
        wm_width, wm_height = int(image_watermark.shape[1] * scale_watermark), int(image_watermark.shape[0] * scale_watermark)
        wm_dim = (wm_width, wm_height)
        image_watermark = cv2.resize(image_watermark, wm_dim, interpolation=cv2.INTER_AREA)

    h_image, w_image, _ = image_combined.shape
    center_y, center_x = int(h_image/2), int(w_image/2)

    h_watermark, w_watermark, _ = image_watermark.shape
    top_y, left_x = center_y - int(h_watermark/2), center_x - int(w_watermark/2)
    bottom_y, right_x = top_y + h_watermark, left_x + w_watermark

    roi = image_combined[top_y:bottom_y, left_x:right_x]
    result = cv2.addWeighted(roi, 1, image_watermark, 0.3, 1)
    image_combined[top_y:bottom_y, left_x:right_x] = result

    return image_combined
    
##############################################################################################################################