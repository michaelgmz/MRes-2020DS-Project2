# %%
# ----------------------------------------------------------------
# There are two types of raw data format, .xlsx and .csv
# These series of scripts deal with .csv only
# All imports data initialized paths goes here
# ----------------------------------------------------------------
import os
import pandas as pd
import re
import dill
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
ROOT_DIR = 'D:\Rotation2\IVM-MRes project #2\IVM machine learning project 2020 (Mingze)'
VIDEO_DIR = 'D:\Rotation2\IVM-MRes project #2\Video data (back-up)'
FRAME_DIR = 'D:\Rotation2\VideoFrame'

# %%
def Get_Obj_Folder(ROOT_DIR, VIDEO_DIR):
    '''
    Find folder path containing .csv files that corresponding to the specific video
    (Videos may contains more frames than .csv files)
    Input: Path of all .csv result folders
           Path of all .avi video folder
    Output: A list of the name of the folders'''

    res = []
    for folder in os.listdir(ROOT_DIR):
        if 'CD11a' in folder and 'B cells' not in folder:
            res.append(folder)
    
    return res

# %%
def Process_Specific_Trial(folder_name):
    '''
    Get relevant characteristics from CSVInfo class
    The function is for a folder which corresponding to a specific trial
    Modify with specific needed.
    A branch is to compares cells deteced by Imaris and (fastER & MRCNN)
    Should be interesting to see if either one can outpeform'''

    name_prefix = os.path.split(folder_name)[1][:-10]
    name_prefix = os.path.join(folder_name, name_prefix)

    variables = CSVInfo()
    position = variables.Position(name_prefix)

    Compare_Imaris(position, name_prefix)

# %%
def Compare_Imaris(item, name_prefix):
    '''
    The function compare cells detected by Imaris and (fastER & MRCNN)
    Overlapping frames will be saved to see either one outweighs
    80 - 140 before
    340 - 400 after'''

    if 'before' in name_prefix:
        frame_folder = 'extract_spleen CD11a blocking 19-4-18 20-160.avi'
        Comparison(frame_folder, [80, 141], 'before', item)

    if 'after' in name_prefix:
        frame_folder = 'extract_spleen CD11a blocking 19-4-18 280-400.avi'
        Comparison(frame_folder, [60, 121], 'after', item)

# %%
def Comparison(frame_folder, interval, comment, item):
    '''
    mrcnnhpc : 'Frame_001.png'
    faster overlap : 'GreyFrame_001_overlap.png
    0.463 um / pixel'''

    fast = os.path.join(FRAME_DIR, frame_folder, 'YellowBlur', 'Overlap')
    cnn = os.path.join(FRAME_DIR, frame_folder, 'YellowBlur', 'mrcnnhpc')
    
    for index in range(interval[0], interval[1]):
        index = str(index).zfill(3)
        cnn_name = 'Frame_' + index + '.png'
        fast_name = 'GreyFrame_' + index + '_overlap.png'

        img_cnn = imread(os.path.join(cnn, cnn_name))
        img_fast = imread(os.path.join(fast, fast_name))

        height, width = img_cnn.shape[0], img_cnn.shape[1]
        img_cnn = img_cnn[15:height - 15, 15:width - 15]
        img_fast = img_fast[15:height - 15, 15:width - 15]
        img_cnn = cv2.resize(img_cnn, (1002, 1002))
        img_fast = cv2.resize(img_fast, (1002, 1002))


# %%
class CSVInfo:
    '''
    Calculate cell info specifically
    Each function correspond to different purporses
    Marked by function name'''

    def Overall(self, name_prefix):
        csv_name = name_prefix + 'Overall.csv'
        csv = pd.read_csv(csv_name, header = 2)
        total_no_cells = csv['Value'][61]

        details = {index:{} for index in range(total_no_cells)}
        for key in details.keys():
            details[key] = {frame:[] for frame in range(1, 62)}

        return details

    def Position(self, name_prefix):
        csv_name = name_prefix + 'Position.csv'
        details = Generate_Dict(csv_name)
        return details

# %%
def Generate_Dict(csv_name):
    '''
    Read content of .csv files and generate a dictionary to store relevant data
    for future easy access
    Dictionary -> Cell Index -> Frame Index -> Value (for 1 value)
                                                X, Y, Z (for 3 values)
    Cell Index starts with 0
    Frame Index starts with 1 1 ~ 61, 61 frames in total'''

    para = {}
    csv = pd.read_csv(csv_name, header = 2)

    for index in range(len(csv)):
        row = csv.iloc[index:index + 1]
        trackid = int(row['TrackID']) - 1000000000
        time = int(row['Time'])

        if not str(trackid) in para.keys():
            para[str(trackid)] = {}

        if not str(time) in para[str(trackid)].keys():
            para[str(trackid)][str(time)] = []

        if csv.columns.values.tolist()[3] == 'Unit':
            X = float(row.iloc[:, 0])
            Y = float(row.iloc[:, 1])
            Z = float(row.iloc[:, 2])
                
            para[str(trackid)][str(time)].append(X)
            para[str(trackid)][str(time)].append(Y)
            para[str(trackid)][str(time)].append(Z)

        if csv.columns.values.tolist()[1] == 'Unit':
            value = float(row.iloc[:, 0])
            para[str(trackid)][str(time)].append(value)
    
    return para