# %%
'''----------------------------------------------------------------
This script creates dictionary data type contains cell info identified by fastER
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
import pickle

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
            for colour in ['Red', 'YellowBlur']:
                seg_list.append(os.path.join(VIDEO_PATH, video, colour, 'segmentation'))
                grey_list.append(os.path.join(VIDEO_PATH, video, colour, 'Grey'))
    
    print (f'--- Video path added ---\n')
    return seg_list, grey_list

# %%
'''----------------------------------------------------------------
Extract individual cell info from a single frame
----------------------------------------------------------------'''
def Cell_Op(seg_list, grey_list):
    '''
    The function calculates average intensity and volume of cells in a frame,
    or across all the frames within one specific video data
    Input: frame folder path contains segmentation folder and greyscale folder 
           required for calculating contour
    Output: to be confirmed '''
    
    dicts = {'volume':[], 'intensity':[], 'circumference':[], 'smooth':[], 
            'contour':[], 'centre':[], 'instances':[]}

    frames_seg = os.listdir(seg_list[0])
    frames_grey = os.listdir(grey_list[0])

    count = 0
    for index in tqdm(range(len(frames_seg))):
        seg_path = os.path.join(seg_list[0], frames_seg[index])
        grey_path = os.path.join(grey_list[0], frames_grey[index])
        img = imread(seg_path)
        img_grey = imread(grey_path)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        threshold_img = img_grey.copy()
        threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

        image_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

        count = count + len(contours)
        dicts['instances'].append(count)
        for contour in contours:
            M = cv2.moments(contour)

            result = Info()
            dicts['contour'].append(contour)
            dicts['volume'].append(result.Volume(M))
            dicts['intensity'].append(result.Intensity(contour, img_grey, image_shape))
            dicts['circumference'].append(result.Circumference(contour))
            smooth, centre = result.Smoothness(M, contour)
            dicts['smooth'].append(smooth)
            dicts['centre'].append(centre)

        # Lines for saving greyscale frame with only cells that are below the threshold volume
        #     if volume[-1] < THRESHOLD:
        #         threshold_img, threshold_dir = Save_Threshold_Frame(threshold_img, grey_list[0], contour)
        # imsave(os.path.join(threshold_dir, frames_grey[index]), threshold_img)

    print (f'--- Cell Info Extracted ---\n')
    return dicts

# %%
'''----------------------------------------------------------------
Check whether small volume identified objects are actually cells or not
----------------------------------------------------------------'''
def Save_Threshold_Frame(threshold_img, grey_path, contour):
    '''
    The function saves the greyscale frame into a specific folder
    Cells with a volume lower than certain threshold are labelled'''
    
    threshold_dir = os.path.join(os.path.split(grey_path)[0], 'Threshold')
    if not os.path.exists(threshold_dir):
        os.mkdir(threshold_dir)
    
    for coordinate in contour:
        peripheral = coordinate[0]
        threshold_img[peripheral[1], peripheral[0], 1] = 255
    
    return threshold_img, threshold_dir

# %%
'''----------------------------------------------------------------
Class type Info. Volume | Intensity | Circumference | Smoothness (1st derivative)
----------------------------------------------------------------'''
class Info:
    '''
    Calculate cell info specifically
    Each function correspond to different purporses
    Marked by function name'''

    def Volume(self, M):
        return M['m00']

    def Intensity(self, contour, img_grey, image_shape):
        copy = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image_shape, dtype = np.uint8)
        colour = [255, 255, 255]
        cv2.fillPoly(mask, [contour], colour)
        res = cv2.bitwise_and(copy, mask)

        pixel = []
        for row in range(np.min(contour, axis = 0)[0][1], np.max(contour, axis = 0)[0][1] + 1):
            for col in range(np.min(contour, axis = 0)[0][0], np.max(contour, axis = 0)[0][0] + 1):
                if res[row, col, 0] != 0:
                    pixel.append(res[row, col, 0])

        pixel_ave = np.mean(pixel)
        return pixel_ave

    def Circumference(self, contour):
        perimeter = cv2.arcLength(contour, True)
        return perimeter
    
    def Smoothness(self, M, contour):
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centre = [cx, cy]

        dst = []
        for coordinate in contour:
            peripheral = coordinate[0]
            dst.append(distance.euclidean(centre, peripheral))
        
        dx = 1
        dy = diff(dst) / dx
        return round(np.mean(abs(dy)), 3), (cx, cy)

# %%
'''----------------------------------------------------------------
Main part of the module goes here
Parameters and information Dictionary initialization
----------------------------------------------------------------'''
if __name__ == '__main__': 
    plt.style.use('ggplot')
    VIDEO_PATH = 'D:\Rotation2\VideoFrame'
    PLOT_PATH = 'D:\Rotation2\Plots'
    IMG_HEIGHT, IMG_WIDTH = 1002, 1002
    IMG_CHANNELS = 3
    THRESHOLD = 150

    seg_folders, grey_folders = Seg_Grey_Path(VIDEO_PATH)

    for item in range(len(seg_folders)):
        if 'Red' in seg_folders[item]:
            print (seg_folders[item])          
            seg_list = []
            grey_list = []
            seg_list.append(seg_folders[item])
            grey_list.append(grey_folders[item])

            if '160' in seg_folders[item]:
                dict_bf = Cell_Op(seg_list, grey_list)
                pickle.dump(dict_bf, open(os.path.join(os.path.split(seg_folders[item])[0], 'dict_bf.pkl'), 'wb'))
            if '280' in seg_folders[item]:
                dict_af = Cell_Op(seg_list, grey_list)
                pickle.dump(dict_af, open(os.path.join(os.path.split(seg_folders[item])[0], 'dict_af.pkl'), 'wb'))

        # if 'Yellow'in seg_folders[item]:     
        #     print (seg_folders[item])          
        #     seg_list = []
        #     grey_list = []
        #     seg_list.append(seg_folders[item])
        #     grey_list.append(grey_folders[item])

        #     if 'before' in seg_folders[item]:
        #         dict_bf = Cell_Op(seg_list, grey_list)
        #         pickle.dump(dict_bf, open(os.path.join(os.path.split(seg_folders[item])[0], 'dict_bf.pkl'), 'wb'))
        #     if 'after' in seg_folders[item]:
        #         dict_af = Cell_Op(seg_list, grey_list)
        #         pickle.dump(dict_af, open(os.path.join(os.path.split(seg_folders[item])[0], 'dict_af.pkl'), 'wb'))