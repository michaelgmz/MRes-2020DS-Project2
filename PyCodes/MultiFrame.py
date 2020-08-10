# %%
'''-------------------------------------------------------------
This script is the (1 / 3) of MOT purpose, next is object_tracker.py
Aims to perform Multi-Object Tracking technique
And generate tracking video with bounding box labels
-------------------------------------------------------------'''
import os 
import cv2
import pickle
import sys
from CellOperation import Seg_Grey_Path
sys.path.append('..')
from objecttrack.object_tracker import Pipeline
import time
import matplotlib.pyplot as plt

# %%
'''----------------------------------------------------------------
Parameter tuning for object tracker pipeline. Video format and size should
be manully defined and in accordance with original raw videos.
----------------------------------------------------------------'''
def Track_Multi(par_path, fps):
    multi_path = os.path.join(par_path, 'multi')
    if not os.path.exists(multi_path):
        os.mkdir(multi_path)
    print ('Parent folder: {}'.format(par_path))

    file_path = os.path.join(multi_path, str(int(time.time()))) + 'fps{}.avi'.format(fps)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    size = (1002, 1002)
    video = cv2.VideoWriter(file_path, fourcc, int(fps), size)

    if '160' in par_path:
        trackers = Pipeline(par_path, multi_path, dict_bm, dict_bdiff, video, 
                    maxDisappeared = 10, write_video = True, label = True, centre = False, 
                    show_img = False, draw_contour = 'Rectangle', kalman = True, min_hits = 3, 
                    save_img = True)
        # plt.imshow(res[0])
        # plt.tight_layout()
        # plt.savefig('D:\Rotation2\VideoFrame\cking.png', format = 'png', dpi = 500)
        pickle.dump(trackers, open(os.path.join(par_path, 'neutrophilbefore.pkl'), 'wb'))
        video.release()

    # if '400' in par_path:
    #     trackers = Pipeline(par_path, multi_path, dict_am, dict_adiff, video, 
    #                 maxDisappeared = 10, write_video = True, label = True, centre = False, 
    #                 show_img = False, draw_contour = 'Rectangle', kalman = True, min_hits = 3)
    #     pickle.dump(trackers, open(os.path.join(par_path, 'neutrophilafter.pkl'), 'wb'))
    #     video.release()

# %%
'''----------------------------------------------------------------
Main part of the module goes here
----------------------------------------------------------------'''
if __name__ == '__main__':
    # VIDEO_PATH = 'D:\Rotation2\VideoFrame'
    # seg_folders, grey_folders = Seg_Grey_Path(VIDEO_PATH)

    # dict_am = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\dict_am.pkl', 'rb'))
    dict_bm = pickle.load(open('D:\Rotation2\VideoFrame\dict_bm.pkl', 'rb'))
    # dict_adiff = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\discrepancy_a.pkl', 'rb'))
    dict_bdiff = pickle.load(open('D:\Rotation2\VideoFrame\discrepancy_b.pkl', 'rb'))

    print ('Please input desired FPS for VideoWriter: ')
    fps = input()
    Track_Multi('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 20-160.avi\YellowBlur', fps)
    # for folder in seg_folders:
    #     par_path = os.path.split(folder)[0]
    #     if 'Red' in folder:
    #         pass
    #     if 'Yellow' in folder:
    #         Track_Multi(par_path, fps)