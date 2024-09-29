# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import numpy as np
import pandas as pd
import os
from ground_truth_object_results.sela_loss_nms_investigation.analyze_spatialzones import divide_scene, flip_axis_to_camera, flip_axis_to_depth, get_scene_dimensions_center, get_zone_for_bbox, get_zones_for_gt_bboxes, plot_scene_go, plot_scene_o3d


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from metric_util import calc_iou # axis-aligned 3D box IoU
def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    #pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d

from box_util import box3d_iou
def get_iou_obb(bb1,bb2):
    iou3d, iou2d = box3d_iou(bb1,bb2)
    return iou3d

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)

def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)

def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
        print(classname, ap[classname])
    
    return rec, prec, ap 

from multiprocessing import Pool
def eval_det_multiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper, [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])
    
    return rec, prec, ap 

# save a scene's gt and pred bboxes to .npy files
def save_gt_pred_bboxes(gt_all, pred_all, dataset, img_id, save_dir):
    gt = gt_all[img_id]
    pred = pred_all[img_id]

    scanname = dataset.scan_names[img_id]

    print('GT and Pred Bboxes for image: ', scanname)
    print("GT: \n", gt)
    print("Pred: \n", pred)

    # Unpacking gt and pred correctly
    try:
        gt_bboxes = [{'class': cls, 'bbox': bbox.tolist()} for cls, bbox in gt]
        pred_bboxes = [{'class': cls, 'bbox': bbox.tolist()} for cls, bbox in pred]
    except ValueError as e:
        print("Error unpacking GT or Pred items:", e)
        # Fallback in case of incorrect structure
        gt_bboxes = [{'class': item[0], 'bbox': item[1].tolist()} for item in gt]
        pred_bboxes = [{'class': item[0], 'bbox': item[1].tolist()} for item in pred]

    # print('Saving gt and pred bboxes to: ', save_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # np.save(os.path.join(save_dir, f'{scanname}_gt.npy'), gt_bboxes)
    # np.save(os.path.join(save_dir, f'{scanname}_pred.npy'), pred_bboxes)
    
def eval_gt_acc(pred_all, gt_all, dataset, img_id_to_check, save_dir, ovthresh=0.25, get_iou_func=get_iou):
    """ Generic functions to check if there is a corresponding predicted bounding box for each ground truth bounding box.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            dataset: Dataset object as defined in /datasets
        Output:
            results: {img_id: [(gt_bbox, pred_bbox, iou)]}
    """
    pred = {} # map {img_id: {classname: [ (bbox, score, idx) ] } }
    gt = {} # map {img_id: {classname: [ bbox, idx ] } }
    
    for img_id in pred_all.keys():
        pred[img_id] = {}
        for idx, (classname, bbox, score) in enumerate(pred_all[img_id]):
            if classname not in pred[img_id]:
                pred[img_id][classname] = []
            pred[img_id][classname].append((bbox, score, idx))
    
    for img_id in gt_all.keys():
        gt[img_id] = {}
        for idx, (classname, bbox) in enumerate(gt_all[img_id]):
            if classname not in gt[img_id]:
                gt[img_id][classname] = []
            gt[img_id][classname].append((bbox, idx))

    results = {} # map {img_id: [gt_bbox, best_pred_bbox, max_iou, gt_bbox_idx, best_pred_bbox_idx] }

    for img_id in gt.keys(): # iterate through each image file
        results[img_id] = []
        
        for classname in gt[img_id].keys(): # iterate all through each class present in the image, according to GT
            gt_boxes = gt[img_id][classname] # get all the GT bboxes of a class in the image
            pred_boxes = pred[img_id].get(classname, []) # get all the predicted bboxes of a class in the image
            
            for (gt_bbox, gt_bbox_idx) in gt_boxes: # iterate through each GT bbox in the image
                max_iou = 0
                best_pred_bbox = None
                best_pred_bbox_idx = None
                
                for (pred_bbox, score, pred_bbox_idx) in pred_boxes: # iterate through each predicted bbox in the image
                    iou = get_iou_func(gt_bbox, pred_bbox) # get the IoU for each pair of bbox for a class between GT and predicted (if any)
                    if iou > ovthresh and iou > max_iou: # check if iou meets the threshold to be accepted, and updates the best predicted bbox if score is highest
                        max_iou = iou
                        best_pred_bbox = pred_bbox
                        best_pred_bbox_idx = pred_bbox_idx
                        
                results[img_id].append((classname, gt_bbox, best_pred_bbox, max_iou, gt_bbox_idx, best_pred_bbox_idx)) # appends result for a GT bbox in the image
    
    data = []
    classnames = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture',
                      'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
    
    for img_id, bboxes in results.items():
        for classname, gt_bbox, pred_bbox, iou, gt_bbox_idx, best_pred_bbox_idx in bboxes:
            data.append({
                'img_id': img_id,
                'scan_name': dataset.scan_names[img_id],
                'classname': classnames[classname],
                'gt_bbox_index': int(gt_bbox_idx),
                'pred_bbox_index':int(best_pred_bbox_idx) if best_pred_bbox_idx is not None else None
            })

        scan_name= dataset.scan_names[img_id]
        gt_bboxes = [{'class': classname, 'bbox': gt_bbox, 'gt_bbox_index': gt_bbox_idx,} for (classname, gt_bbox, best_pred_bbox, iou, gt_bbox_idx, best_pred_bbox_idx) in bboxes]
        pred_bboxes = [{'class': classname, 'bbox': pred_bbox, 'gt_bbox_index': gt_bbox_idx, 'pred_bbox_index': pred_bbox_idx,} for idx, (classname, gt_bbox, best_pred_bbox, iou, gt_bbox_idx, best_pred_bbox_idx) in enumerate(bboxes)]
        '''
        print('Saving gt and pred bboxes to: ', save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, f'{scan_name}_gt.npy'), gt_bboxes)
        np.save(os.path.join(save_dir, f'{scan_name}_pred.npy'), pred_bboxes)
        '''
    df = pd.DataFrame(data, columns=['img_id', 'scan_name', 'classname', 'gt_bbox_index', 'pred_bbox_index'])
    return df

def eval_det_cls_wrapper_task(args):
    zone_index, classname, pred_data, gt_data, ovthresh, use_07_metric, get_iou_func = args
    rec_zone_cls, prec_zone_cls, ap_zone_cls = eval_det_cls(
        pred_data, gt_data, ovthresh, use_07_metric, get_iou_func
    )
    return (zone_index, classname, rec_zone_cls, prec_zone_cls, ap_zone_cls)

def eval_det_spatialzones_multiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou, n=4):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {spatialzone_idx: rec}
            prec: {spatialzone_idx: prec_all}
            ap: {spatialzone_idx: scalar}
    """
    pred = {} # map {spatialzone_idx: pred}
    gt = {} # map {spatialzone_idx: gt}
    
    # Get the spatial zones for the GT bboxes
    gt_zone_mappings = {} # Dictionary( img_id: Dictionary( bbox_index: (classname, zone_index) ) )
    pred_zone_mappings = {} # Dictionary( img_id: Dictionary( bbox_index: (classname, zone_index) ) )
    
    for img_id in gt_all.keys():
        gt_zone_mappings[img_id] = get_zones_for_gt_bboxes(img_id, n_zones=n)
        for bbox_index, (classname, bbox) in enumerate(gt_all[img_id]):
            zone_index = gt_zone_mappings[img_id][bbox_index][1]
            gt_all[img_id][bbox_index] = (classname, bbox, zone_index)
        
        for classname, bbox, zone_index in gt_all[img_id]:
            gt.setdefault(zone_index, {}).setdefault(classname, {}).setdefault(img_id, []).append(bbox)
            
            
    for img_id in pred_all.keys():
        (scene_dimensions, scene_center, gt_entities) = get_scene_dimensions_center(img_id)
        zones = divide_scene(n, scene_dimensions, scene_center)
        
        for bbox_index, (classname, bbox, score) in enumerate(pred_all[img_id]):
            bbox_center = np.mean(bbox, axis=0)
            zone_index = get_zone_for_bbox(bbox_center=bbox_center, zones=zones)
            '''
            if zone_index == -1:
                print('Zone index is -1 for bbox: ', bbox)
                print('Scene center: ', scene_center)
                print('Scene dimensions: ', scene_dimensions)
                print('Zones: ', zones)
                print('Bbox center: ', bbox_center)
                print('Bbox: ', bbox)
                print('Img_id: ', img_id)
                plot_scene_go(zones, [bbox_center], gt_entities)
                break
            '''
            pred_all[img_id][bbox_index] = (classname, bbox, score, zone_index)
        
        for classname, bbox, score, zone_index in pred_all[img_id]:
            pred.setdefault(zone_index, {}).setdefault(classname, {}).setdefault(img_id, []).append((bbox, score))
    

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    tasks = []

    for zone_index in gt.keys():
        for classname in gt[zone_index]:
            gt_data = gt[zone_index][classname]
            pred_data = pred.get(zone_index, {}).get(classname, {})
            tasks.append((
                zone_index, classname, pred_data, gt_data,
                ovthresh, use_07_metric, get_iou_func
            ))

    results = p.map(eval_det_cls_wrapper_task, tasks)
    p.close()
    p.join()

    for zone_index, classname, rec_zone_cls, prec_zone_cls, ap_zone_cls in results:
        rec.setdefault(zone_index, {})[classname] = rec_zone_cls
        prec.setdefault(zone_index, {})[classname] = prec_zone_cls
        ap.setdefault(zone_index, {})[classname] = ap_zone_cls
        print(zone_index, classname, ap_zone_cls)

    # Handle zone_indices not in pred by setting metrics to zero
    for zone_index in gt.keys():
        for classname in gt[zone_index]:
            if classname not in pred.get(zone_index, {}):
                rec.setdefault(zone_index, {})[classname] = np.array([]) 
                prec.setdefault(zone_index, {})[classname] = np.array([])
                ap.setdefault(zone_index, {})[classname] = 0.0
                print(zone_index, classname, 0.0)

    return rec, prec, ap


def eval_det_spatialzones(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou, n=4):
    """Compute precision/recall for object detection across multiple spatial zones and classes.

    Args:
        pred_all (dict): Map of {img_id: [(classname, bbox, score)]}
        gt_all (dict): Map of {img_id: [(classname, bbox)]}
        ovthresh (float): IoU threshold.
        use_07_metric (bool): If True, use VOC07 11 point method.
        get_iou_func (function): Function to compute IoU.
        n (int): Number of spatial zones.

    Returns:
        rec (dict): {spatialzone_idx: {classname: rec_array}}
        prec (dict): {spatialzone_idx: {classname: prec_array}}
        ap (dict): {spatialzone_idx: {classname: ap_value}}
    """
    pred = {}  # Map {zone_index: {classname: {img_id: [(bbox, score), ...]}}}
    gt = {}    # Map {zone_index: {classname: {img_id: [bbox, ...]}}}
    
    gt_zone_mappings = {} # Dictionary( img_id: Dictionary( bbox_index: (classname, zone_index) ) )
    pred_zone_mappings = {} # Dictionary( img_id: Dictionary( bbox_index: (classname, zone_index) ) )
    

    # Process ground truth data
    for img_id in gt_all.keys():
        gt_zone_mappings[img_id] = get_zones_for_gt_bboxes(img_id, n_zones=n)
        for bbox_index, (classname, bbox) in enumerate(gt_all[img_id]):
            zone_index = gt_zone_mappings[img_id][bbox_index][1]
            gt_all[img_id][bbox_index] = (classname, bbox, zone_index)

        for classname, bbox, zone_index in gt_all[img_id]:
            gt.setdefault(zone_index, {}).setdefault(classname, {}).setdefault(img_id, []).append(bbox)

    # Process prediction data
    for img_id in pred_all.keys():
        (scene_dimensions, scene_center, gt_entities) = get_scene_dimensions_center(img_id)
        zones = divide_scene(n, scene_dimensions, scene_center)

        for bbox_index, (classname, bbox, score) in enumerate(pred_all[img_id]):
            bbox_camera = flip_axis_to_camera(bbox)
            # bbox_depth = flip_axis_to_depth(bbox_camera)
            # bbox_depth = flip_axis_to_depth(bbox_depth)
            bbox_center = np.mean(bbox_camera, axis=0)
            
            zone_index = get_zone_for_bbox(bbox_center=bbox_center, zones=zones)
            
            if zone_index == -1:
                print('Zone index is -1 for bbox: ', bbox_camera)
                print('Scene center: ', scene_center)
                print('Scene dimensions: ', scene_dimensions)
                print('Zones: ', zones)
                print('Bbox center: ', bbox_center)
                print('Bbox: ', bbox_camera)
                print('Img_id: ', img_id)
                plot_scene_go(zones, [bbox_camera], gt_entities)
                break
            
            pred_all[img_id][bbox_index] = (classname, bbox, score, zone_index)

        for classname, bbox, score, zone_index in pred_all[img_id]:
            pred.setdefault(zone_index, {}).setdefault(classname, {}).setdefault(img_id, []).append((bbox, score))

    rec = {}
    prec = {}
    ap = {}

    # Sequentially evaluate each zone and class
    for zone_index in gt.keys():
        for classname in gt[zone_index]:
            gt_data = gt[zone_index][classname]
            pred_data = pred.get(zone_index, {}).get(classname, {})
            rec_zone_cls, prec_zone_cls, ap_zone_cls = eval_det_cls(
                pred_data, gt_data, ovthresh=ovthresh,
                use_07_metric=use_07_metric, get_iou_func=get_iou_func
            )
            rec.setdefault(zone_index, {})[classname] = rec_zone_cls
            prec.setdefault(zone_index, {})[classname] = prec_zone_cls
            ap.setdefault(zone_index, {})[classname] = ap_zone_cls
            print(zone_index, classname, ap_zone_cls)

    # Handle classes not present in predictions by setting metrics to zero
    for zone_index in gt.keys():
        for classname in gt[zone_index]:
            if classname not in pred.get(zone_index, {}):
                rec.setdefault(zone_index, {})[classname] = np.array([])
                prec.setdefault(zone_index, {})[classname] = np.array([])
                ap.setdefault(zone_index, {})[classname] = 0.0
                print(zone_index, classname, 0.0)

    return rec, prec, ap
