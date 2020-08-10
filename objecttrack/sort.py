# %%
'''----------------------------------------------------------------
This script is the (3 / 3) of MOT purpose, previous is object_tracker.py
MOT | KalmanFilter | Distance based Data Association Methods
{
Acknowledgement: SIMPLE ONLINE AND REALTIME TRACKING
Paper Author: Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben
}
----------------------------------------------------------------'''
from scipy.spatial import distance as dist
import numpy as np
from filterpy.kalman import KalmanFilter

# %%
'''----------------------------------------------------------------
Numba Just-in-time compiler for python to speed up algorithms
----------------------------------------------------------------'''
try:
    from numba import jit
except:
    def jit(func):
        return func
np.random.seed(0)

# %%
'''----------------------------------------------------------------
Data Association Hungarian Algorithm (Provided using IOU as tracking strategy)
----------------------------------------------------------------'''
def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i >= 0])

    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

# %%
'''----------------------------------------------------------------
Computes IUO between two bboxes in the form [x1, y1, x2, y2]
----------------------------------------------------------------'''
@jit
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)

    return (o)

# %%
'''----------------------------------------------------------------
Convert bounding box to centre scale
----------------------------------------------------------------'''
def convert_bbox_to_z(bbox):
    '''
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area 
        and r is the aspect ratio
    '''
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.

    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

# %%
'''----------------------------------------------------------------
Convert centre scale to bounding box
----------------------------------------------------------------'''
def convert_x_to_bbox(x, score = None):
    '''
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    '''
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

# %%
'''----------------------------------------------------------------
Kalman Filter initialization | Update Stage | Predict Stage
----------------------------------------------------------------'''
class KalmanBoxTracker(object):
    '''
    This class represents the internal state of individual tracked objects observed as bbox.
    '''
    count = 0

    def __init__(self, bbox, frame_count, pattern, cont):
        '''
        Initialises a tracker using initial bounding box.
        '''
        #define constant velocity model
        self.kf = KalmanFilter(dim_x = 7, dim_z = 4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], 
                            [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], 
                            [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                            [0,0,0,1,0,0,0]])

        self.kf.R[2:, 2:] *= 1.

        #give low uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1. 
        self.kf.P *= 1.

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = set()
        self.insert = set()
        self.id.add(KalmanBoxTracker.count)
        self.insert.add(frame_count)
        KalmanBoxTracker.count += 1

        self.history_pre = []
        self.history_upd = {frame_count: convert_bbox_to_z(bbox)}
        self.hits = 0

        self.pat = {frame_count: pattern}
        self.cont = {frame_count: cont}

    def update(self, bbox, frame_count, pattern, cont):
        '''
        Updates the state vector with observed bbox.
        '''
        self.time_since_update = 0
        self.history_pre = []
        self.history_upd[frame_count] = convert_bbox_to_z(bbox)
        self.hits += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.pat[frame_count] = pattern
        self.cont[frame_count] = cont

    def update_id(self, new_id, frame_count):
        if new_id not in self.id:
            self.id.add(new_id)
            self.insert.add(frame_count)

    def predict(self):
        '''
        Advances the state vector and returns the predicted bounding box estimate.
        '''
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.time_since_update += 1
        self.history_pre.append(convert_x_to_bbox(self.kf.x))
            
        return self.history_pre[-1]

    def get_cont(self):
        return self.cont

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

# %%
'''----------------------------------------------------------------
Detection Tracker Association for subsequent frames
Distance based (Centroid Tracking Algorithm) & IOU based methods
----------------------------------------------------------------'''
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3, 
    kfilter = None):
    '''
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    '''

    if (len(trackers) == 0):
        return np.empty((0, 2), dtype = int), np.arange(len(detections)), np.empty((0, 5), dtype = int)
    
    else:
        if iou_threshold == 0:
            trackCentroids = []
            inputCentroids = np.zeros((len(detections), 2), dtype = "int")

            # inputCentroids[i]
            for i, position in enumerate(detections):
                cX = int((position[0] + position[2]) / 2.0)
                cY = int((position[1] + position[3]) / 2.0)
                inputCentroids[i] = (cX, cY)

            # trackCentroids[i]
            for i, trk in enumerate(trackers):
                cX = int((trk[0] + trk[2]) / 2.0)
                cY = int((trk[1] + trk[3]) / 2.0)
                trackCentroids.append((cX, cY))
            
            D = dist.cdist(np.array(trackCentroids), inputCentroids, metric = 'euclidean')
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]
            usedRows = set()
            usedCols = set()

            # matched_indices[inputIndex, trackIndex]
            matched_indices = []
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                usedRows.add(row)
                usedCols.add(col)
                matched_indices.append(np.array([col, row]))

            unmatched_trackers = list(set(range(0, D.shape[0])).difference(usedRows))
            unmatched_detections = list(set(range(0, D.shape[1])).difference(usedCols))

            matches = []
            for m in matched_indices:
                trajectory = kfilter[m[1]].history_upd
                aveg_dist, last_dist, max_dist = compute_aveg_dist(trajectory)

                info = kfilter[m[1]].pat
                dis_threshold = np.mean([info[key][3] for key in info.keys()]) / 2

                if D[m[1], m[0]] <= dis_threshold:
                    matches.append(m.reshape(1, 2))
                else:
                    unmatched_trackers.append(m[1])
                    unmatched_detections.append(m[0])

        # use IOU as tracking strategy
        else:
            iou_matrix = np.zeros((len(detections), len(trackers)), dtype = np.float32)
            for d, det in enumerate(detections):
                for t, trk in enumerate(trackers):
                    iou_matrix[d, t] = iou(det, trk)

            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis = 1)
                else:
                    matched_indices = linear_assignment(-iou_matrix)
            else:
                matched_indices = np.empty(shape = (0, 2))

            unmatched_detections = []
            for d, det in enumerate(detections):
                if (d not in matched_indices[:, 0]):
                    unmatched_detections.append(d)

            unmatched_trackers = []
            for t, trk in enumerate(trackers):
                if (t not in matched_indices[:, 1]):
                    unmatched_trackers.append(t)

            #filter out matched with low IOU
            matches = []
            for m in matched_indices:
                if (iou_matrix[m[0], m[1]] < iou_threshold):
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])

                else:
                    matches.append(m.reshape(1, 2))
                
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype = int)
        else:
            matches = np.concatenate(matches, axis = 0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# %%
'''----------------------------------------------------------------
Computer average distance of a trajectory
This function is currently REDUNDENT
----------------------------------------------------------------'''
def compute_aveg_dist(trajectory):
    centre = []
    distances = []
    for n, frame_index in enumerate(trajectory.keys()):
        centre.append((trajectory[frame_index][0], trajectory[frame_index][1]))

        if len(centre) == 1:
            index = frame_index
        else:
            distances.append(dist.euclidean(centre[-1], centre[-2]))

    if len(trajectory) == 1:
        return 0, 0, 0
    else:
        return sum(distances[:]) / (frame_index - index), distances[-1], \
            max(distances)

# %%
'''----------------------------------------------------------------
Main part of the tracking algorithm goes here
Parameters can be tuned in MultiFrame.py
----------------------------------------------------------------'''
class SortMot(object):
    def __init__(self, max_age = 1, min_hits = 3, iou_threshold = 0.3):
        '''
        Sets key parameters for SORT
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.trackers_resi = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.total_obj = 0
        self.pop_obj = 0
        self.dets_pattern = {0: None}

    def update(self, dets = np.empty((0, 5)), pattern = None, conts = None):
        '''
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections 
        (use np.empty((0, 5)) for frames without detections).
        Returns: a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        '''

        self.frame_count += 1
        self.dets_pattern[self.frame_count] = pattern
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        ret_cont = []

        # predicted locations for next frame trk[x1, y1, x2, y2]
        for t, trk in enumerate(trks):
            position = self.trackers[t].predict()[0]
            trk[:] = [position[0], position[1], position[2], position[3], 0]

            if np.any(np.isnan(position)):
                to_del.append(t)

        # post processing trackers suppress invalid values
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            self.pop_obj += 1

        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(dets, trks, iou_threshold = self.iou_threshold, 
                kfilter = self.trackers)
    
        # update matched trackers with assigned detections
        # matched[0] is detected, matched[1] is tracked
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], self.frame_count, 
                self.dets_pattern[self.frame_count][m[0]], conts[m[0]])

        '''
        # create and initialise new trackers for unmatched detections
        unmatched_dets = unmatched_dets.tolist()
        for i in unmatched_dets:
            centroid = self.dets_pattern[self.frame_count][i][4]
            for trk in self.trackers:
                if len(trk.id) > 1:
                    if dist.euclidean(centroid, trk.pat[max(trk.pat.keys())][4]) < \
                        trk.pat[max(trk.pat.keys())][3] / 2:
                        trk.id.remove(min(trk.id))
                        unmatched_dets.remove(i)
                        break
        '''
            
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], self.frame_count, 
                    self.dets_pattern[self.frame_count][i], conts[i])
            self.trackers.append(trk)

        # update unmatched trackers to assign multiple ids
        for m in matched:
            bbox = dets[m[0], 0:4]
            for trk in self.trackers:
                centroid = trk.pat[max(trk.pat.keys())][4]
                identity = min(trk.id)

                if centroid[0] > bbox[0] and centroid[0] < bbox[2] and \
                    centroid[1] > bbox[1] and centroid[1] < bbox[3]:
                    self.trackers[m[1]].update_id(identity, self.frame_count)

        # add valid tracklet according to parameters
        # and remove dead tracklet
        i = len(self.trackers)
        usedids = set()
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            c = trk.get_cont()

            '''
            if len(trk.id) > 1:
                if min(trk.id) in usedids:
                    trk.id.remove(min(trk.id))
            '''

            if min(trk.id) not in usedids:
                if (trk.time_since_update < self.max_age) and \
                    (trk.hits >= self.min_hits or \
                    self.frame_count <= self.min_hits):
                    ret.append(np.concatenate((d, [min(trk.id) + 1])).reshape(1, -1))
                    usedids.add(min(trk.id))
                    ret_cont.append(c)

            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers_resi.append(trk)
                self.trackers.pop(i)
                self.pop_obj += 1

        if (len(ret) > 0):
            return np.concatenate(ret), self.pop_obj, self.trackers_resi, self.trackers

        return np.empty((0, 5)), self.pop_obj, self.trackers_resi, self.trackers
        