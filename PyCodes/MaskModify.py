# %%
'''----------------------------------------------------------------
This script exports canny and segmentation mask from fastER results
Then overlap canny pic and greyscale pic to generate fastER preview
All imports and pre-defined parameters go here
----------------------------------------------------------------'''
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow, imsave
import dill
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
VIDEO_PATH = 'D:\Rotation2\VideoFrame'
IMG_HEIGHT, IMG_WIDTH = 1002, 1002
IMG_CHANNELS = 3

# %%
'''----------------------------------------------------------------
Get frame folder path for each video, colour, greyscale frame
& segmentation results
----------------------------------------------------------------'''
def Seg_Grey_Path(VIDEO_PATH):
    '''
    Input: Head of video frame paths
    Output: list of segmentatin folder & list of greyscale folder'''
    seg_list = []
    grey_list = []

    for video in os.listdir(VIDEO_PATH):
        if video.startswith('extract'):
            # if '280-400' not in video and '20-160' not in video:
                for colour in ['Red', 'YellowBlur']:
                    seg_list.append(os.path.join(VIDEO_PATH, video, colour, 'segmentation'))
                    grey_list.append(os.path.join(VIDEO_PATH, video, colour, 'Grey'))
    
    print (f'--- Video path added ---\n')
    return seg_list, grey_list

# %%
'''----------------------------------------------------------------
Export segmentation mask from fastER results
Detect the canny of each mask and store them accordingly
----------------------------------------------------------------'''
def Export_Canny(seg_list):
    '''
    Input: list of segmentation folder
    Output: canny for every frame and video in numpy format
            segmentation for every frame and video in numpy format'''
    canny_total = []
    # seg_total = []

    for folder in range(len(seg_list)):
        segs = os.listdir(seg_list[folder])
        canny = np.zeros((len(segs), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
        # seg = np.zeros((len(segs), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

        canny_dir = os.path.join(os.path.split(seg_list[folder])[0], 'Canny')
        if not os.path.exists(canny_dir):
            os.mkdir(canny_dir)

        for n, frame in tqdm(enumerate(segs)):
            img = imread(os.path.join(seg_list[folder], frame))
            img = cv2.Canny(img, 50, 100)
            canny[n, :, :, 2] = img
            canny[n] = cv2.cvtColor(canny[n], cv2.COLOR_BGR2RGB)
            canny_name = frame[:-12] + 'canny.png'
            plt.imsave(os.path.join(canny_dir, canny_name) + '.png', canny[n], format = 'png')

        # seg_total.append(seg)
        canny_total.append(canny)
        print (f'--- Canny folder # [{folder}] out of [{len(seg_list) - 1}] ---\n')
    return canny_total

# %%
'''----------------------------------------------------------------
Overlap canny pictiure with greyscale picture to generate fastER preview results
Then we should be able to calculate average intensity or volume of cells
----------------------------------------------------------------'''
def Export_Overlap(grey_list, canny_total):
    '''
    Input: list of greyscale folder
           canny for every frame and video in numpy format
    Output: overlap for every frame and video in numpy format'''
    # overlap_total = []
    
    for folder in range(len(grey_list)):
        canny_to_use = canny_total[folder]
        greys = os.listdir(grey_list[folder])
        overlap = np.zeros((len(greys), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

        overlap_dir = os.path.join(os.path.split(grey_list[folder])[0], 'Overlap')
        if not os.path.exists(overlap_dir):
            os.mkdir(overlap_dir)
        
        for n, frame in tqdm(enumerate(greys)):
            img = imread(os.path.join(grey_list[folder], frame))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for row in range(IMG_HEIGHT):
                for col in range(IMG_WIDTH):
                    if canny_to_use[n][row, col, 0] != 0:
                        img[row, col, 2] = canny_to_use[n][row, col, 0]

            overlap[n] = img
            overlap[n] = cv2.cvtColor(overlap[n], cv2.COLOR_BGR2RGB)
            overlap_name = frame[:-4] + '_overlap'
            plt.imsave(os.path.join(overlap_dir, overlap_name) + '.svg', overlap[n], format = 'svg')
            plt.imsave(os.path.join(overlap_dir, overlap_name) + '.png', overlap[n], format = 'png')
        
        # overlap_total.append(overlap)
        print (f'--- Overlap folder # [{folder}] out of [{len(grey_list) - 1}] ---\n')
    # return overlap_total

# %%
'''----------------------------------------------------------------
Main part of the module goes here
----------------------------------------------------------------'''
if __name__ == '__main__':
    seg_folders, grey_folders = Seg_Grey_Path(VIDEO_PATH)

    contours = []
    for item in range(len(seg_folders)):
        if 'Red' in seg_folders[item]:
            pass

        if 'Yellow'in seg_folders[item]:
            print (seg_folders[item])
            seg_list = []
            grey_list = []
            seg_list.append(seg_folders[item])
            grey_list.append(grey_folders[item])

            canny_total = Export_Canny(seg_list)
            Export_Overlap(grey_list, canny_total)

    os.chdir(VIDEO_PATH)