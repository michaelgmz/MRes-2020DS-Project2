# %%
'''----------------------------------------------------------------
This script is the (2 / 3) of MOT purpose, previous is MultiFrame.py, next is sort.py
Aims to combine Mask-RCNN and discrepancy results and pass to tracker
----------------------------------------------------------------'''
import sys
sys.path.append('..')
from objecttrack.centroidtracker import CentroidTracker
import numpy as np
import cv2
import os
from skimage.io import imread, imshow
from tqdm import tqdm
import objecttrack.sort as sort
import matplotlib.pyplot as plt

# %%
'''----------------------------------------------------------------
Pipeline. Parameters can be tuned according to feasible options in MultiFrame.py
Threshold is arbitrarily specified. Change when needed.
----------------------------------------------------------------'''
def Pipeline(par_path, multi_path, dict_m, dict_diff, video, maxDisappeared = 3, 
			write_video = False, label = False, centre = False, show_img = False, 
			draw_contour = 'None', kalman = False, min_hits = 1, iou_threshold = 0, 
            save_img = False):
			
	# initialize centroidtracker class and get available frames
	if kalman:
		mot = sort.SortMot(max_age = maxDisappeared, min_hits = min_hits, iou_threshold = iou_threshold)
	else:
		ct = CentroidTracker(maxDisappeared = maxDisappeared)
	
	(H, W) = (None, None)
	frames = os.listdir(os.path.join(par_path, 'Colour'))

	assert draw_contour in ['Circle', 'Rectangle', 'None'], 'Contour shape not supported'

	pattern_total = {}
	total_obj = 0
	for frame_count, img_name in enumerate(tqdm(frames)):
		# read image and acquire dimensions
		frame = imread(os.path.join(par_path, 'Colour', img_name))
		(H, W) = (np.shape(frame)[0], np.shape(frame)[1])
		
		pattern = {}
		conts = []
		if kalman:
			rects = np.empty((0, 5), dtype = int)
		else:
			rects = []

		# set index for manipulating frames to be used in dict_m
		index = int(img_name[6:9])
		if index == 1:
			start_m, end_m = 0, dict_m['instances'][0]
		else:
			start_m, end_m = dict_m['instances'][index - 2], dict_m['instances'][index - 1]

		# add bbox from MRCNN results set threshold for each frame or entire dataset
		volume_thres = np.percentile(dict_m['volume'], 7.5)
		intensity_thres = np.percentile(dict_m['intensity'], 2.5)

		# 'intensity' 'volume' 'smooth' 'circumference'
		valid = 0
		for m, contour in enumerate(dict_m['contour'][start_m:end_m]):
			if dict_m['volume'][start_m + m] > volume_thres and \
				dict_m['intensity'][start_m + m] > intensity_thres:

				x, y, w, h = cv2.boundingRect(contour)
				(startX, startY, endX, endY) = x, y, x + w, y + h

				pattern[valid] = []
				pattern[valid].append(dict_m['intensity'][start_m + m])
				pattern[valid].append(dict_m['volume'][start_m + m])
				pattern[valid].append(dict_m['smooth'][start_m + m])
				pattern[valid].append(dict_m['circumference'][start_m + m])
				pattern[valid].append(dict_m['centre'][start_m + m])
				pattern[valid].append((startX, startY, endX, endY))
				valid += 1

				if kalman:
					rects = np.concatenate((rects, np.array([[startX, startY, endX, endY, 1]])))
					conts.append(contour)
				else:
					rects.append((startX, startY, endX, endY))

				if draw_contour == 'Circle':
					cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
				elif draw_contour == 'Rectangle':
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
				else:
					pass

		# add bbox from fastER results (exclude MRCNN result)
		if str(index) in dict_diff.keys():
			for n, contour in enumerate(dict_diff[str(index)]['contour']):
				x, y, w, h = cv2.boundingRect(contour)
				(startX, startY, endX, endY) = x, y, x + w, y + h

				pattern[valid] = []
				pattern[valid].append(dict_diff[str(index)]['intensity'][n])
				pattern[valid].append(dict_diff[str(index)]['volume'][n])
				pattern[valid].append(dict_diff[str(index)]['smooth'][n])
				pattern[valid].append(dict_diff[str(index)]['circumference'][n])
				pattern[valid].append(dict_diff[str(index)]['centre'][n])
				pattern[valid].append((startX, startY, endX, endY))
				valid += 1

				if kalman:
					rects = np.concatenate((rects, np.array([[startX, startY, endX, endY, 1]])))
					conts.append(contour)
				else:
					rects.append((startX, startY, endX, endY))
					
				if draw_contour == 'Circle':
					cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
				elif draw_contour == 'Rectangle':
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
				else:
					pass

		# update our centroid tracker using the computed set of bbox rectangles
		if kalman:
			objects, pop_obj, trackers_resi, trackers = mot.update(rects, pattern, conts)
		else:
			objects = ct.update(rects)

		# loop over the tracked objects
		if kalman:
			for n, obj in enumerate(objects):
				text = "{}".format(int(obj[-1]))

				if label:
					cv2.putText(frame, text, (int((obj[0] + obj[2]) / 2), int((obj[1] + obj[3]) / 2)),
								cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1)

		else:
			for (objectID, centroid) in objects.items():
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "{}".format(objectID)
				if label:
					cv2.putText(frame, text, (centroid[0] - 10, centroid[1] + 5),
								cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1)
				if centre:
					cv2.circle(frame, (centroid[0], centroid[1]), 2, (255, 255, 255), -1)

		# show the output frame for debugging purpose
		if show_img:
			cv2.imshow("Frame {}".format(str(index)), frame)
			key = cv2.waitKey(0) & 0xFF
			if key == ord("q"):
				break
			cv2.destroyAllWindows()
		
		if write_video:
			frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
			video.write(frame)

	if kalman:
		print (len(trackers))
		print (len(trackers_resi))
		trackers.extend(trackers_resi)
		for trk in trackers:
			if trk.hits >= min_hits:
				total_obj += 1
		print ('Total # of tracked objects: {}'.format(total_obj))
		print ('Total # of poped objects: {}\n'.format(pop_obj))
	else:
		print ('Total # of tracked objects: {}'.format(ct.nextObjectID - 1))
		print ('Total # of deregistered objects: {}\n'.format(len(ct.deregistered.keys())))
	
	return trackers