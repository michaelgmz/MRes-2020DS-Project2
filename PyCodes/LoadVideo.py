# %%
'''----------------------------------------------------------------
Video formatt is .avi Primarily use OpenCV to process data
All imports data initialized paths goes here
Function 1. Read and Extract all the frames from videos for downstream
          cell segmentation
----------------------------------------------------------------'''
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
'''----------------------------------------------------------------
Capture Video and get each frames then save them into .png format
Each Video will be in a different folder
----------------------------------------------------------------'''
def CaptureVideo(video_path):
    video_name = os.path.split(video_path)[1]
    print (f'----- {video_name} -----')
    os.chdir('D:\Rotation2\VideoFrame')
    frame_folder = './' + 'original_' + video_name
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

    cap = cv2.VideoCapture(video_path)
    total_frame = cap.get(7)
    print (f'----- Total Number of Frames [{cap.get(7)}] -----')

    Index = 1
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            print (f'----- Frame # {Index} -----')
            frame = Crop_Pic(frame)
            ExtractColor(img = frame, video_name = video_name, index = Index)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            plt.xticks([])
            plt.yticks([])
            plt.title('Frame ' + str(Index))
            os.chdir('D:\Rotation2\VideoFrame')
            plt.imsave(frame_folder + '/Frame_' + str(Index) + '.png', img_rgb, format = 'png')
            plt.close()
        
        Index = Index + 1
        if Index > total_frame:
            break

    cap.release()
    cv2.destroyAllWindows()

# %%
'''----------------------------------------------------------------
Crop the image to exclude the 50um and timelines
Assume it is a squre and it locates at the centre of the pics
----------------------------------------------------------------'''
def Crop_Pic(img):
    shape = img.shape
    height, width = shape[0], shape[1]
    left_edge = int(width / 2 - height / 2)
    right_edge = int(width / 2 + height / 2)
    img = img[0:height, left_edge:right_edge]
    return img

# %%
'''----------------------------------------------------------------
Extrace Red and Yellow Color Channel
----------------------------------------------------------------'''
def ExtractColor(img, video_name, index):
    '''
    For yellow part, set red and blue channel to 0, therefore convert them to green channel only
    For red part, set blue channel to 0
    For the rest of the place, substract Red - Green to get pure red
    Two types of cells combined. Then substract Red - Green again to acquire pure B cells
    Then adjust the negative or noise values
    '''
    os.chdir('D:\Rotation2\VideoFrame')
    frame_folder = './' + 'extract_' + video_name
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

    yellow = img.copy()
    yellow[:, :, 0] = 0 # blue
    yellow[:, :, 2] = 0 # red
    SaveImg(yellow, 'Yellow', frame_folder, index)

    blur = yellow.copy()
    blur = cv2.pyrMeanShiftFiltering(blur, 25, 5)
    os.chdir('D:\Rotation2\VideoFrame')
    SaveImg(blur, 'YellowBlur', frame_folder, index)

    red = img.copy()
    red[:, :, 0] = 0 # blue
    red[:, :, 1] = 0 # green

    for i in range(np.shape(red)[0]):
        for j in range(np.shape(red)[1]):
            if red[i, j, 2] < yellow[i, j ,1]:
                red[i, j, 2] = 0
            else:
                red[i, j, 2] = red[i, j, 2] - yellow[i, j, 1]

    os.chdir('D:\Rotation2\VideoFrame')
    SaveImg(red, 'Red', frame_folder, index)

# %%
'''----------------------------------------------------------------
Save individual image to Red and Yellow folder respectively
----------------------------------------------------------------'''
def SaveImg(img, color, frame_folder, index):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not os.path.exists(frame_folder + '/' + color):
        os.mkdir(frame_folder + '/' + color)
    os.chdir(frame_folder + '/' + color)

    for contrast in ['Colour', 'Grey']:
        if not os.path.exists('./' + contrast):
            os.mkdir('./' + contrast)
        if contrast == 'Colour':
            plt.imsave('./' + contrast + '/Frame_' + str(index).zfill(3) + '.png', img, format = 'png')
        if contrast == 'Grey':
            cv2.imwrite('./' + contrast + '/GreyFrame_' + str(index).zfill(3) + '.png', img_grey)

# %%
'''----------------------------------------------------------------
Get all the videos paths from objective_directory
----------------------------------------------------------------'''
def Get_Video_Path(objective_directory):
    all_files = []
    for lists in os.listdir(objective_directory):
        if lists.endswith('.avi'):
            all_files.append(os.path.join(objective_directory, lists))
    return all_files

# %%
'''----------------------------------------------------------------
Main part of the module goes here
----------------------------------------------------------------'''
if __name__ == '__main__':
    # A series of videos go here.
    objective_directory = 'D:\Rotation2\IVM-MRes project #2\Video data (back-up)'
    all_files = Get_Video_Path(objective_directory)
    for files in all_files:
        CaptureVideo(files)
    
    # A solely video goes here.
    CaptureVideo('D:\Rotation2\VideoFrame\Exp 18-5-18 FTY720\FTY720 spleen 1h both cells after.avi')