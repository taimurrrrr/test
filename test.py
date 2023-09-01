import torch
import numpy as np
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter
import cv2

def convert_gt_ici5(bboxes):
    vertices = []
    for idx,bbox in enumerate(bboxes):
        bbox = np.float32(bbox)
        print(bbox)
        rect = cv2.minAreaRect(bbox)  # 最小外接矩形
        print(bbox)
        box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整

        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.max(box[:, 1])
        bottom_point_y = np.min(box[:, 1])
        left_point_y = sorted(box[:, 1][np.where(box[:, 0] == left_point_x)])[-1]
        right_point_y = sorted(box[:, 1][np.where(box[:, 0] == right_point_x)])[0]
        top_point_x = sorted(box[:, 0][np.where(box[:, 1] == top_point_y)])[-1]
        bottom_point_x = sorted(box[:, 0][np.where(box[:, 1] == bottom_point_y)])[0]

        vertice = [[left_point_x, left_point_y], [bottom_point_x, bottom_point_y],
                    [right_point_x, right_point_y], [top_point_x, top_point_y]]
        vertices.append(np.array(vertice))

    return vertices


def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pse_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        # image = data['imgs'].data.cpu().numpy()
        # image = (image* 255).astype(np.uint8)

        data['imgs'] = data['imgs'].cuda()

        data.update(dict(
            cfg=cfg
        ))
        # if idx == 1:
        #     idx=0
        #     break
        # forward
        with torch.no_grad():
            outputs = model(**data)
            # contours = convert_gt_ici5(outputs['bboxes'])
            # print(contours)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        # print('image_name', image_name)
        rf.write_result(image_name, outputs)



def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    # print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
