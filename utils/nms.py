# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from pc_util import bbox_corner_dist_measure

# boxes are axis aigned 2D boxes of shape (n,5) in FLOAT numbers with (x1,y1,x2,y2,score)
''' Ref: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
Ref: https://github.com/vickyboy47/nms-python/blob/master/nms.py 
'''
def nms_2d(boxes, overlap_threshold):
    # print("Using nms_2d")
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    # sort bounding boxes by confidence scores
    I = np.argsort(score)

    # initialize the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes list
    while (I.size!=0):
        last = I.size
        
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        suppress = [last-1] # add the index to the list of suppressed indexes (i.e., indexes that will be removed)
        for pos in range(last-1): # iterate through all possible bounding boxes
            j = I[pos] # get the current index
            xx1 = max(x1[i],x1[j])
            yy1 = max(y1[i],y1[j])
            xx2 = min(x2[i],x2[j])
            yy2 = min(y2[i],y2[j])
            w = xx2-xx1
            h = yy2-yy1
            if (w>0 and h>0):
                o = w*h/area[j] # IoU score
                print('Overlap is', o)
                if (o>overlap_threshold):
                    suppress.append(pos) # add the index to the list of suppressed indexes (i.e., indexes that will be removed)
        I = np.delete(I,suppress) # delete all indexes from the indexes list that are in the suppress list
    return pick # return the list of picked indexes

def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    # print("Using nms_2d_faster")
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        if old_type: # old type of nms
            o = (w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms
            inter = w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0]))) # delete all indexes from the indexes list that are in the suppress list

    return pick

def nms_2d_faster_sela(boxes, overlap_threshold, alpha, old_type=False, gamma=1):
    # print("Using nms_2d_faster_sela")
    
    thresholds = overlap_threshold - gamma * alpha
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    area = (x2-x1)*(y2-y1)

    I = np.argsort(score)   # I is the index of the sorted scores, from low to high
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1] # i is the index of the box with the highest confidence score
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        if old_type: # old type of nms
            o = (w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms:
            inter = w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score

        I = np.delete(I, np.concatenate(([last-1], np.where(o>thresholds[0][i])))) # delete all indexes from the indexes list that are in the suppress list

    return pick

def nms_3d_faster(boxes, overlap_threshold, old_type=False):  # Used by SDCoT
    # print("Using nms_3d_faster")
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    # sort bounding boxes by confidence scores
    I = np.argsort(score)

    # initialize the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes list
    while (I.size!=0):
        last = I.size 
        
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1]  # index of the box with the highest score
        pick.append(i)

        # find the largest (x, y, z) coordinates for the start of the bounding box and the smallest (x, y, z) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        # compute the width, height, and depth of the bounding box
        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type: # old type of nms
            o = (l*w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0]))) # delete all indexes from the indexes list that are in the suppress list

    return pick

def nms_3d_faster_sela(boxes, overlap_threshold, alpha, old_type=False, gamma=1):  # Used by SDCoT
    # print("Using nms_3d_faster_sela")
    thresholds = overlap_threshold - gamma * alpha
    
    # print("shape of thresholds: ", thresholds.shape)
    # print("shape of boxes: ", boxes.shape)
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    # sort bounding boxes by confidence scores
    I = np.argsort(score)

    # initialize the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes list
    while (I.size!=0):
        last = I.size 
        
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1]  # index of the box with the highest score
        pick.append(i)

        # find the largest (x, y, z) coordinates for the start of the bounding box and the smallest (x, y, z) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        # compute the width, height, and depth of the bounding box
        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type: # old type of nms
            o = (l*w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score
            
        thresholds = overlap_threshold - gamma * alpha[I[:last - 1]]
        I = np.delete(I, np.concatenate(([last-1], np.where(o>thresholds)[0]))) # delete all indexes from the indexes list that are in the suppress list

    return pick


def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):    # Used during basetraining
    # print("Using nms_3d_faster_samecls")
    
    # print("Shape of boxes is", boxes.shape)
    #print("Shape of overlap_threshold is", overlap_threshold.shape)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7] # class label
    area = (x2-x1)*(y2-y1)*(z2-z1)

    # sort bounding boxes by confidence scores
    I = np.argsort(score)
    # initialize the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes list
    while (I.size!=0):
        
        last = I.size 
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1] # index of the box with the highest score
        pick.append(i)

        # find the largest (x, y, z) coordinates for the start of the bounding box and the smallest (x, y, z) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        # get the class labels of the boxes
        cls1 = cls[i]   
        cls2 = cls[I[:last-1]]

        # compute the width, height, and depth of the bounding box
        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type: # old type of nms
            o = (l*w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score

        o = o * (cls1==cls2) # only consider the boxes with the same class label

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0]))) # delete all indexes from the indexes list that are in the suppress list

    return pick


def nms_3d_faster_samecls_sela(boxes, overlap_threshold, alpha, old_type=False, gamma=0.1):
    # print("Using nms_3d_faster_samecls_sela")
    
    thresholds = overlap_threshold + gamma * alpha
    # thresholds is a 1D array of shape (n,) where n is the number of boxes
    
    # print("Shape of boxes is", boxes.shape)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7] # class label
    area = (x2-x1)*(y2-y1)*(z2-z1)

    # sort bounding boxes by confidence scores
    I = np.argsort(score)
    # initialize the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes list
    while (I.size!=0):
        
        last = I.size 
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1] # index of the box with the highest score
        pick.append(i)

        # find the largest (x, y, z) coordinates for the start of the bounding box and the smallest (x, y, z) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        # get the class labels of the boxes
        cls1 = cls[i]   
        cls2 = cls[I[:last-1]]

        # compute the width, height, and depth of the bounding box
        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type: # old type of nms
            o = (l*w*h)/area[I[:last-1]] # IoU score
        else: # new type of nms
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter) # IoU score

        o = o * (cls1==cls2) # only consider the boxes with the same class label

        # print("thresholds[i]: ", thresholds[i])
        # print("np.where(o>thresholds[i]) : ", np.where(o>thresholds[i]))
        # print("np.where(o>thresholds[i])[0] : ", np.where(o>thresholds[i])[0])
        I = np.delete(I, np.concatenate(([last-1], np.where(o>thresholds[i])[0]))) # delete all indexes from the indexes list that are in the suppress list

    return pick


def nms_crnr_dist(boxes, conf, overlap_threshold):
    # print("Using nms_crnr_dist")   
    I = np.argsort(conf) # sort bounding boxes by confidence scores
    pick = [] # initialize the list of picked indexes
    while (I.size!=0): # keep looping while some indexes still remain in the indexes list
        last = I.size

        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = I[-1] 
        pick.append(i)        
        
        scores = []

        # find the corner distance between the bounding boxes
        for ind in I[:-1]:
            scores.append(bbox_corner_dist_measure(boxes[i,:], boxes[ind, :]))
        
        # remove the bounding boxes with corner distance less than the threshold
        I = np.delete(I, np.concatenate(([last-1], np.where(np.array(scores)>overlap_threshold)[0])))

    return pick

if __name__=='__main__':
    a = np.random.random((100,5))
    print(nms_2d(a,0.9))
    print(nms_2d_faster(a,0.9))
