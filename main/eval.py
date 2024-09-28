""" Evaluation phase for incremental 3D object detection.

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.backends
import torch.backends.cudnn
from torch.utils.data import DataLoader, SequentialSampler
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'trainers'))
from opts import parse_args
from logger import init_logger
from model import create_detection_model, load_detection_model, generate_pseudo_bboxes
from loss_helper import get_supervised_loss
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from train_bt import my_worker_init_fn


def evaluate(args, model, dataloader, logger, device, dataset_config, dataset):
    logger.cprint(str(datetime.now()))
    
    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': args.use_3d_nms,
                   'nms_iou': args.nms_iou, 'use_old_type_nms': args.use_old_type_nms,
                   'cls_nms': args.use_cls_nms, 'per_class_proposal': args.per_class_proposal,
                   'conf_thresh': float(args.conf_thresh), 'dataset_config': dataset_config,
                   'obj_conf_thresh': 0.95, 'cls_conf_thresh': 0.95}
    
    ap_calculator = APCalculator(float(args.ap_iou_threshold), dataset_config.class2type)

    stat_dict = {}
    model.eval()  # set model to eval mode (for bn and dp)
    classifier_weights_learned = torch.empty_like(model.prediction_header.classifier_weights).copy_(
                                        model.prediction_header.classifier_weights.detach())
    classifier_weights_learned = classifier_weights_learned.squeeze(-1)

    logger.cprint(f"------------ **AFTER** Classifier Weights Learned:------------  \n {classifier_weights_learned}")
    
    for batch_idx, batch_data_label in enumerate(dataloader):

        if batch_idx % 50 == 0:
            print('Eval batch: %d' % (batch_idx))

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = model(batch_data_label['point_clouds'])

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = get_supervised_loss(end_points, dataset_config, sela=False)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # log statstics
    for key in sorted(stat_dict.keys()):
        logger.cprint('eval mean %s: %f' % (key, stat_dict[key] / float(batch_idx + 1)))

    # Evaluate average precision and per-object accuracy
    # Pass in the dataset so that we can map the img_id to the scan name
    # metrics_dict, gt_df = ap_calculator.compute_metrics(dataset)
    metrics_dict, gt_df, sz_metrics_dict = ap_calculator.compute_metrics(dataset)
    for key in metrics_dict:
        logger.cprint('eval %s: %f' % (key, metrics_dict[key]))

    logger.cprint(f"------------ Ground Truths Dataframe: ------------: \n")
    # Print each row of gt_df
    # for index, row in gt_df.iterrows():
    #     logger.cprint(str(row))

    # Save DataFrame as CSV
    csv_file_path = f'gt_df_results_{str(datetime.now())}.csv'
    gt_df.to_csv(csv_file_path, index=True)
    
    # logger.cprint(f"------------ Saving pseudo bboxes: ------------: \n")
    # point_cloud = dataset[46]['point_clouds']
    # scan_name = dataset.scan_names[46]
    
    # pseudo_bboxes = generate_pseudo_bboxes(model, device, CONFIG_DICT, point_cloud)
    
    # # Save pseudo_bboxes
    # save_dir = "pseudo_bboxes_npy"
    # logger.cprint('Saving gt and pred bboxes to: ' + save_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # np.save(os.path.join(save_dir, f'{scan_name}_pseudo_bboxes_0.95.npy'), pseudo_bboxes)

    # Save the sz_metrics_dict to a CSV file
    print(sz_metrics_dict)
    sz_metrics_dict_df = pd.DataFrame.from_dict(sz_metrics_dict, orient='index')
    sz_metrics_dict_df.reset_index(inplace=True)
    sz_metrics_dict_df.rename(columns={'index': 'zone_index'}, inplace=True)
    print(sz_metrics_dict_df)

    sz_metrics_dict_df.to_csv(f'sz_metrics_dict_{str(datetime.now())}.csv', index=[0])

def main(args):
    logger = init_logger(args)
    
    # Initialise seed to control randomness:
    
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)
    
    
    # ======== Init Dataset =========
    if args.method == 'basetrain':
        EVAL_ALL_CLASS = False
    else:
        EVAL_ALL_CLASS = True

    if args.dataset == 'sunrgbd':
        from sunrgbd_val import SunrgbdValDataset
        test_dataset = SunrgbdValDataset(all_classes=EVAL_ALL_CLASS,
                                         num_novel_class=args.n_novel_class,
                                         num_points=args.num_point,
                                         use_color=args.use_color,
                                         use_height=(not args.no_height),
                                         augment=False)

    elif args.dataset == 'scannet':
        from scannet_val import ScannetValDataset
        test_dataset = ScannetValDataset(all_classes=EVAL_ALL_CLASS,
                                         num_novel_class=args.n_novel_class,
                                         num_points=args.num_point,
                                         use_color=args.use_color,
                                         use_height=(not args.no_height),
                                         augment=False)
        print(test_dataset.scan_names)
        
        from scannet_base import ScannetBaseDataset
        train_dataset = ScannetBaseDataset(num_points=args.num_point,
                                           use_color=args.use_color,
                                           use_height=(not args.no_height),
                                           augment=args.pc_augm)
    else:
        print('Unknown dataset %s. Exiting...' % (args.dataset))
        exit(-1)
    
    print("Batch Size used: ", args.batch_size)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.batch_size//2, worker_init_fn=my_worker_init_fn,)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.batch_size//2, worker_init_fn=my_worker_init_fn,)
    
    TEST_DATASET_CONFIG = test_dataset.dataset_config
    TRAIN_DATASET_CONFIG = train_dataset.dataset_config

    model = create_detection_model(args, TEST_DATASET_CONFIG)
    classifier_weights_BEFORE = torch.empty_like(model.prediction_header.classifier_weights).copy_(
                                        model.prediction_header.classifier_weights.detach())
    classifier_weights_BEFORE = classifier_weights_BEFORE.squeeze(-1)

    logger.cprint(f"------------ **BEFORE** Classifier Weights Learned:------------  \n {classifier_weights_BEFORE}")
    if args.model_checkpoint_path is not None:
        if args.method == 'finetune':
            model_checkpoint = torch.load(os.path.join(ROOT_DIR, args.model_checkpoint_path, 'checkpoint.tar'))
            base_classifier_weights = np.load(os.path.join(ROOT_DIR, args.model_checkpoint_path,
                                                                 'base_classifier_weights.npy'))
            base_classifier_weights = torch.from_numpy(base_classifier_weights).cuda()
            loaded_model_state_dict = model_checkpoint['model_state_dict']
            model_state_dict = model.state_dict()

            for k in loaded_model_state_dict:
                if k in model_state_dict:
                    if loaded_model_state_dict[k].shape != model_state_dict[k].shape:
                        print('For %s, concatenate base and novel classifier weights...' %k)
                        assert loaded_model_state_dict[k].shape[0] + base_classifier_weights.shape[0] == \
                               model_state_dict[k].shape[0]
                        loaded_model_state_dict[k] = torch.cat((loaded_model_state_dict[k], base_classifier_weights),
                                                                dim=0)
                else:
                    print('Drop parameter {}.'.format(k))

            for k in model_state_dict:
                if not (k in loaded_model_state_dict):
                    print('No param {}.'.format(k))
                    loaded_model_state_dict[k] = model_state_dict[k]

            model.load_state_dict(loaded_model_state_dict, strict=True)

        else:
            model = load_detection_model(model, args.model_checkpoint_path, model_name=args.model_name)
    else:
        raise ValueError('Detection model checkpoint path must be given!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    
    np.random.seed(seed)
    evaluate(args, model, test_dataloader, logger, device, TEST_DATASET_CONFIG, test_dataset)


if __name__ == '__main__':
    args = parse_args()
    main(args)