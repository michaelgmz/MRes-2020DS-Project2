#%%
'''----------------------------------------------------------------
This script shows the interaction cell as video format
----------------------------------------------------------------'''
import pickle
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
import sys
sys.path.append('..')
import objecttrack
import time
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 1002, 1002, 3
N_PATH = 'D:\Rotation2\VideoFrame\SecondCD11a'
B_PATH = 'D:\Rotation2\VideoFrame\Exp 19-4-18 CD11a blocking'
ORI_Be_PATH = 'D:\Rotation2\VideoFrame\original_spleen CD11a blocking 19-4-18 before both cells video frames 80-140.avi'
ORI_Af_PATH = 'D:\Rotation2\VideoFrame\original_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi'

'''----------------------------------------------------------------
self.id, set, may contain multiple ids
self.hits, int, # of frames the cell has been hit
self.pat, dict, {frame # : [intensity, volume, smooth, circumference, centre, 
    (startX, startY, endX, endY), velocity, acceleration, displacement]}
self.cont, np.array, {frame # : contour}, contour of the detected cell
----------------------------------------------------------------'''
inter_before = pickle.load(open(os.path.join(B_PATH, 'interbefore.pkl'), 'rb'))
inter_after = pickle.load(open(os.path.join(B_PATH, 'interafter.pkl'), 'rb'))
N_before = pickle.load(open(os.path.join(N_PATH, 'ncellbefore.pkl'), 'rb'))
N_after = pickle.load(open(os.path.join(N_PATH, 'ncellafter.pkl'), 'rb'))

#%%
'''----------------------------------------------------------------
Read B cells information | ExcelFile data
----------------------------------------------------------------'''
def ReadBFile(B_PATH, label = None):
    for file in os.listdir(B_PATH):
        if file.endswith('.xlsx') and 'B' in file:
            if label in file:
                B = pd.ExcelFile(os.path.join(B_PATH, file))

    return B

#%%
def TrackInter(ORI_PATH, fps, label = None):
    inter_path = os.path.join(ORI_PATH, 'TrackInter')
    if not os.path.exists(inter_path):
        os.mkdir(inter_path)
    print (f'---> Interaction tracking folder located at {os.path.split(ORI_PATH)[1]}')

    file_path = os.path.join(inter_path, str(int(time.time()))) + 'fps{}.avi'.format(fps)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    size = (1002, 1002)
    video = cv2.VideoWriter(file_path, fourcc, int(fps), size)
    video = Pipeline(ORI_PATH, video, label)
    video.release()

#%%
def Pipeline(ORI_PATH, video, label = None):
    B = ReadBFile(B_PATH, label)
    b_position = pd.read_excel(B, sheet_name = 'Position', header = 1)
    inter_count = 0

    if label == 'before':
        N_info = N_before
        df_inter = inter_before
        b_position['Position X'] = b_position['Position X'] * 1.7105 - 983.6
        b_position['Position Y'] = b_position['Position Y'] * (-1.7098) + 1955.9
    if label == 'after':
        N_info = N_after
        df_inter = inter_after
        b_position['Position X'] = b_position['Position X'] * 1.7395 - 1007.4
        b_position['Position Y'] = b_position['Position Y'] * (-1.7216) + 1964.7
    
    frames = os.listdir(ORI_PATH)

    for wrong_count, img_name in enumerate(frames):
        if img_name.endswith('.png'):
            print (f'---> {img_name}')
            frame = imread(os.path.join(ORI_PATH, img_name))
            right_count = int(img_name[6:-4])
            # print (right_count)

            b_frame_pos = b_position[b_position['Time'] == right_count]
            for n, item in enumerate(df_inter[str(right_count)]):
                n_id = df_inter.index[n]

                if item != [] and item != -1:
                    b_ids = item
                    for cell in N_info:
                        if len(cell.id) == 1:
                            if list(cell.id)[0] == n_id:
                                n_contour = cell.cont
                                n_cont = n_contour[right_count]
                                frame = cv2.drawContours(frame, [n_cont], -1, (255, 255, 255), 1)

                    for b_id in b_ids:
                        inter_count += 1
                        trackid = b_id + 1000000000
                        for index, row in b_frame_pos.iterrows():
                            if row['TrackID'] == trackid:
                                b_centre = (int(row['Position X']), int(row['Position Y']))
                                frame = cv2.circle(frame, b_centre, int(4 / 0.481), (255, 255, 255), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            video.write(frame)

    print (f'Total # of interactions: {inter_count}')
    return video

#%%
TrackInter(ORI_Be_PATH, fps = 3, label = 'before')
TrackInter(ORI_Af_PATH, fps = 3, label = 'after')