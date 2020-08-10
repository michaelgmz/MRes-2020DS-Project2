# %%
'''----------------------------------------------------------------
This script combines all the frames available
The result will be an active area for Neutrophils and B cells 
And regions where not a single cell goes
Relevant imports and pre-defined parameters listed here
----------------------------------------------------------------'''
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %%
'''----------------------------------------------------------------
Function to combine frames and generate Active & Spot Nothing regions
----------------------------------------------------------------'''
def combine_frame(original_folder, background):
    if not os.path.exists(os.path.join(original_folder, 'filter')):
        os.mkdir(os.path.join(original_folder, 'filter'))
    print (f'---> Filter folder in {os.path.split(original_folder)[1]}')

    frames = os.listdir(original_folder)
    for n, frame in tqdm(enumerate(frames)):
        if frame.endswith('.png'):
            img = cv2.imread(os.path.join(original_folder, frame))
            kernel = np.ones((5, 5), dtype = np.uint8)
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            img_copy = opening.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            # Tune threshold according to previous analysis
            opening[opening < 30] = 0 
            background = cv2.add(opening, background)

    return background

# %%
'''----------------------------------------------------------------
Main part of the module goes here
----------------------------------------------------------------'''
if __name__ == '__main__':
    VIDEO_PATH = 'D:\Rotation2\VideoFrame'
    IMG_HEIGHT, IMG_WIDTH = 1002, 1002
    bg_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype = np.uint8)

    for item in os.listdir(VIDEO_PATH):
        item_path = os.path.join(VIDEO_PATH, item)

        if item.startswith('original') and os.path.isdir(item_path):
            img_combine = combine_frame(item_path, bg_img)
            plt.imsave(item_path[:-4] + '.svg', img_combine, format = 'svg')
            plt.imsave(item_path[:-4] + '.png', img_combine, format = 'png')