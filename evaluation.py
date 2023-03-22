#!/usr/bin/env python
# encoding: utf-8

import os
import time
import glob
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse

from model import SBTNet


def read_meta_data(meta_file_path):
    if not os.path.isfile(meta_file_path):
        raise ValueError(f"Meta file missing under {meta_file_path}.")
    meta = {}
    with open(meta_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
        meta[id] = (src_lens, tgt_lens, disparity)
    return meta


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--root_folder', required=True, type=str)
parser.add_argument('--save_folder', required=True, type=str)
parser.add_argument('--pretrained_path', type=str, default='checkpoints/model.pth')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

src_paths = sorted(glob.glob(os.path.join(args.root_folder, "*.src.jpg")))
meta_data = read_meta_data(os.path.join(args.root_folder, "meta.txt"))

os.makedirs(args.save_folder, exist_ok=True)

model = SBTNet().cuda()
model = torch.nn.DataParallel(model)
checkpoint = torch.load(args.pretrained_path)
model.load_state_dict(checkpoint, strict=True)
model.eval()

print("start processing")

duration = 0

with torch.no_grad():
    for i in tqdm(range(len(src_paths))):
        torch.cuda.empty_cache()

        src_path = src_paths[i]
        if os.path.isdir(src_path):
            continue

        src = cv2.imread(src_path).astype(np.float32)[..., ::-1] / 255
        src = torch.from_numpy(src).permute(2, 0, 1)

        filename = os.path.basename(src_path)
        id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = meta_data[id]
        src_lens_type = torch.tensor(len(src_lens.split('50mmf')[0]) - 4, dtype=torch.float32)  # Sony: 0, Canon: 1
        tgt_lens_type = torch.tensor(len(tgt_lens.split('50mmf')[0]) - 4, dtype=torch.float32)  # Sony: 0, Canon: 1
        src_F = torch.tensor(float(src_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        tgt_F = torch.tensor(float(tgt_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        disparity = torch.tensor(float(disparity), dtype=torch.float32) / 100

        h, w = src.shape[1:]
        cateye_x, cateye_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-h / w, h / w, h))
        cateye_coord = np.concatenate([cateye_x[..., None], cateye_y[..., None]], axis=-1).astype(np.float32)
        cateye_coord = torch.from_numpy(cateye_coord).permute(2, 0, 1)

        src = src[None].cuda()
        src_lens_type = src_lens_type[None].cuda()
        tgt_lens_type = tgt_lens_type[None].cuda()
        src_F = src_F[None].cuda()
        tgt_F = tgt_F[None].cuda()
        disparity = disparity[None].cuda()
        cateye_coord = cateye_coord[None].cuda()

        torch.cuda.synchronize()
        t0 = time.time()

        h_pad = (h + 95) // 96 * 96
        w_pad = (w + 95) // 96 * 96
        src = F.pad(src, pad=(0, w_pad - w, 0, h_pad - h), mode='replicate')
        cateye_coord = F.pad(cateye_coord, pad=(0, w_pad - w, 0, h_pad - h), mode='replicate')

        pred, pred_alpha = model(src, src_lens_type, tgt_lens_type, src_F, tgt_F, disparity, cateye_coord)

        pred = pred[..., :h, :w].clamp(0, 1)
        pred_alpha = pred_alpha[..., :h, :w]

        torch.cuda.synchronize()
        t1 = time.time()
        duration += (t1 - t0)

        pred = pred[0].permute(1, 2, 0).detach().cpu().numpy()

        save_path = os.path.join(args.save_folder, os.path.basename(src_path).replace('.src', '.src_out'))
        cv2.imwrite(save_path, pred[..., ::-1] * 255)  # , [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print('runtime per image[s]:', duration / len(src_paths))
print("finished")
