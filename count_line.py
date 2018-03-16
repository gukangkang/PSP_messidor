import cv2
import scipy.misc as misc
import numpy as np
import os

disc_threshold = 40


def extract_disc_contour(image):
    ret, binary = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


disc_threshold = 45

psp_folder = '/home/imed/workspace/GU/performance_messidor/unet'

mask_folder = '/home/imed/workspace/GU/performance_messidor/unet'
origin_folder = 'save_image'
save_folder = 'performance'

for image in os.listdir(psp_folder):
    mask_image_name = image
    print(image)
    image_path = os.path.join(mask_folder, mask_image_name)
    mask_image = misc.imread(image_path)

    # origin_image_name = 'AGLAIA_GT_' + image[:3] + '.jpg'
    origin_image_name = image

    origin_path = os.path.join(origin_folder, origin_image_name)
    origin_image = misc.imread(origin_path)

    base = np.zeros(shape=np.shape(mask_image))
    base[mask_image > disc_threshold] = 1

    lesion_disc_contour = extract_disc_contour(base.astype(np.uint8))
    disc_labeled_image = cv2.drawContours(origin_image, lesion_disc_contour, -1, (0, 255, 0), 4)
    misc.imsave(os.path.join(save_folder, image), disc_labeled_image)