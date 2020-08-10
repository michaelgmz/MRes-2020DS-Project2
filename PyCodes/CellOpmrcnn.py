# %%
'''----------------------------------------------------------------
This script creates dictionary data type contains cell info identified by Mask-RCNN
Volume | Intensity | Circumference | Smooth | Contour | Centre | Instances
Process individual frames respectively. All imports go here
----------------------------------------------------------------'''
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import diff
from skimage.io import imread, imshow, imsave
import seaborn as sns
from tqdm import tqdm
from scipy.spatial import distance
from CellOperation import Save_Threshold_Frame, Info, Seg_Grey_Path
import pickle

# %%
'''----------------------------------------------------------------
Extract individual cell info from a single frame
----------------------------------------------------------------'''
def Extract_Frame(frame, folder, dicts, count):
    '''
    This function pipelines of processing images are as followes
    1. Generate greyscale, threshold image has below certain volume
    2. Generate sementation image with contour directly drawn in a binary format
    3. Calculate volume, intensity, circumference and smoothness of each contour within
       one frame'''

    img = imread(frame) # (R, G, B)
    
    # Generate greyscale image
    img_grey = img.copy()
    img_grey = cv2.cvtColor(img_grey, cv2.COLOR_RGB2GRAY)

    # Generate threshold image
    threshold_img = img_grey.copy()
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

    # Generate segmentation image, anywhere R is 0, all channels are 0
    # Results would be all the circles with filled red colour
    # Then convert image to binary format
    for row in range(IMG_HEIGHT):
        for col in range(IMG_WIDTH):
            if img[row, col, 0] != 0:
                img[row, col, 0] = 255
                img[row, col, 1] = 255
                img[row, col, 2] = 255
            else:
                img[row, col, 1] = 0
                img[row, col, 2] = 0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find contours, outer contours is the one we need
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # count = count + len(contours)

    for n, h in enumerate(hierarchy[0]):
        if h[2] == -1:
            count = count + 1
            M = cv2.moments(contours[n])

            result = Info()
            dicts['contour'].append(contours[n])
            dicts['volume'].append(result.Volume(M))
            dicts['intensity'].append(result.Intensity(contours[n], img_grey, image_shape))
            dicts['circumference'].append(result.Circumference(contours[n]))
            smooth, centre = result.Smoothness(M, contours[n])
            dicts['smooth'].append(smooth)
            dicts['centre'].append(centre)
    dicts['instances'].append(count)
    
    # Lines for saving greyscale frame with only cells that are below the threshold volume
    #     if volume[-1] < THRESHOLD:
    #         threshold_img, threshold_dir = Save_Threshold_Frame(threshold_img, folder, contour)
    # imsave(os.path.join(threshold_dir, os.path.split(frame)[1]), threshold_img)

    return count

# %%
'''----------------------------------------------------------------
Intermediate function. Pass single frame each time callling the function
----------------------------------------------------------------'''
def Save_Info(dicts, folder):
    mrcnn = os.path.join(os.path.split(folder)[0], 'mrcnn')
    count = 0
    
    for frame in tqdm(os.listdir(mrcnn)):
        if frame.endswith('.png'):
            count = Extract_Frame(os.path.join(mrcnn, frame), folder, dicts, count)

# %%
'''----------------------------------------------------------------
Main part of the module goes here
Parameters and information Dictionary initialization
----------------------------------------------------------------'''
if __name__ == '__main__': 
    plt.style.use('ggplot')
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 1002, 1002, 3
    image_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    VIDEO_PATH = 'D:\Rotation2\VideoFrame'
    THRESHOLD = 150

    seg_list, grey_list = Seg_Grey_Path(VIDEO_PATH)

    for n, folder in enumerate(seg_list):
        if 'Yellow' in folder:
            print (folder)

            if 'before' in folder:
                dict_bm = {'volume':[], 'intensity':[], 'circumference':[], 'smooth':[], 
                    'contour':[], 'centre':[], 'instances':[]}
                Save_Info(dict_bm, folder)
                pickle.dump(dict_bm, open(os.path.join(os.path.split(folder)[0], 'dict_bm.pkl'), 'wb'))
            if 'after' in folder:
                dict_am = {'volume':[], 'intensity':[], 'circumference':[], 'smooth':[], 
                    'contour':[], 'centre':[], 'instances':[]}
                Save_Info(dict_am, folder)
                pickle.dump(dict_am, open(os.path.join(os.path.split(folder)[0], 'dict_am.pkl'), 'wb'))

    #         if '20-160' in folder:
    #             dict_bm = {'volume':[], 'intensity':[], 'circumference':[], 'smooth':[], 
    #                     'contour':[], 'centre':[], 'instances':[]}
    #             Save_Info(dict_bm, folder)

    #         else:
    #             dict_am = {'volume':[], 'intensity':[], 'circumference':[], 'smooth':[], 
    #                     'contour':[], 'centre':[], 'instances':[]}
    #             Save_Info(dict_am, folder)

    # pickle.dump(dict_bm, open(os.path.join(VIDEO_PATH, 'dict_bm.pkl'), 'wb'))
    # pickle.dump(dict_am, open(os.path.join(VIDEO_PATH, 'dict_am.pkl'), 'wb'))