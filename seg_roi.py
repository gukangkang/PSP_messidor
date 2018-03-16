import scipy.misc as misc
import scipy.io as io
import os
import numpy as np
image_path = '/home/imed/Data/Messidor/result_from_XGPC/Messidor/bwmask/20051019_38557_0100_PP.jpg'
mask_path = '/home/imed/Data/Messidor/result_from_XGPC/Messidor/bwmask/20051019_38557_0100_PP.mat'

image = misc.imread(image_path)
print('The image shape is {}'.format(np.shape(image)))


mask = io.loadmat(mask_path)['gtmask']
print('The mask shape is {}'.format(np.shape(mask)))

image_folder = '/home/imed/Data/Messidor/result_from_XGPC/Messidor/im'
mask_folder = '/home/imed/Data/Messidor/result_from_XGPC/Messidor/bwmask'

save_image_roi = 'save_image'
save_mask_roi = 'save_mask'

if not os.path.exists(save_image_roi):
    os.mkdir(save_image_roi)

if not os.path.exists(save_mask_roi):
    os.mkdir(save_mask_roi)



# write all image name to the txt name
# image_folder = '/home/imed/Data/Messidor/result from XGPC/Messidor/bwmask'
# all_images = os.listdir(image_folder)
# all_files = open('filename.txt', 'w')
#
# for image in all_images[:]:
#     print(image[-3:])
#     if image[-3:] == 'jpg':
#         all_files.writelines(image+'\n')

file_txt = 'center_point.txt'
all_names = open(file_txt).readlines()
for list in all_names[:]:
    # print(image_name[:-5])
    # 20051130_54333_0400_PP
    name = list.split(',')[0]
    # x_point = float(list.split(',')[1])
    # y_point = float(list.split(',')[2])
    length = 224
    center_point = [float(list.split(',')[1]), float(list.split(',')[2])]

    image_name = os.path.join(image_folder, name+'.jpg')
    mask_name = os.path.join(mask_folder, name+'.mat')

    # image = misc.imresize(misc.imread(image_name), (960, 1440))
    image = misc.imread(image_name)
    image_shape = np.shape(image)
    mask = io.loadmat(mask_name)['gtmask']

    cropped_image = image[
                    int(center_point[0] * image_shape[0]) - length: int(center_point[0] * image_shape[0]) + length,
                    int(center_point[1] * image_shape[1]) - length: int(center_point[1] * image_shape[1]) + length,
                    :]

    cropped_mask = mask[
                    int(center_point[0] * image_shape[0]) - length: int(center_point[0] * image_shape[0]) + length,
                    int(center_point[1] * image_shape[1]) - length: int(center_point[1] * image_shape[1]) + length,
                    ]

    misc.imsave(os.path.join(save_image_roi, name+'.png'), cropped_image)
    misc.imsave(os.path.join(save_mask_roi, name+'.png'), cropped_mask)






'''
version
header
platform
gtmask
'''

for i in range(0, 256):
    if i in mask:
        print(i)

# print(np.max(mask), np.min(mask))
