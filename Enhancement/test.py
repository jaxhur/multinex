# Copyright (c) 2026 Alexandru Brateanu
# Multinex is licensed for non-commercial research and educational use only.
# Commercial use requires prior written permission.
# See LICENSE for details.


import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ast import arg
import csv
import logging
import numpy as np
import os
import argparse
import shlex
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics
from skimage.color import rgb2ycbcr  # <-- NEW

from basicsr.models import create_model
from basicsr.complexity import _compute_complexity, _count_params
from basicsr.utils.options import dict2str, parse

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(description='Image enhancement evaluation for Multinex')
parser.add_argument('--input_dir', default='./Enhancement/Datasets', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./test_result', type=str, help='Root directory for test results')
parser.add_argument('--output_dir', default='', type=str, help='Directory for output')
parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
parser.add_argument('--weights', type=str, required=True,
                    help='Path to a trained *_G.pth checkpoint')
parser.add_argument('--dataset', default='SDSD_indoor', type=str, help='Test Dataset')
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')
parser.add_argument('--complexity_size', default='256x256', type=str,
                    help='HxW input used for Params/MACs/FLOPs reporting')
args = parser.parse_args()

# Use stable dataset names for result directories while keeping the original
# command-line identifiers required by the dataset dispatch below.
dataset_dir_names = {
    'LOL_v1': 'LOL-v1',
    'LOL_v2_real': 'LOL-v2-real',
    'LOL_v2_synthetic': 'LOL-v2-syn',
}
dataset = args.dataset
dataset_dir = dataset_dir_names.get(dataset, dataset)
dataset_root = os.path.join(args.result_dir, dataset_dir)
result_dir = args.output_dir or os.path.join(dataset_root, 'enhanced')
result_dir_input = os.path.join(dataset_root, 'input')
result_dir_gt = os.path.join(dataset_root, 'gt')
os.makedirs(dataset_root, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

test_logger = logging.getLogger('multinex_test')
test_logger.setLevel(logging.INFO)
test_logger.propagate = False
test_logger.handlers.clear()
test_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
for handler in (
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(dataset_root, 'test.log'), mode='w', encoding='utf-8')):
    handler.setFormatter(test_formatter)
    test_logger.addHandler(handler)
test_logger.info('Command: %s', shlex.join(sys.argv))

# GPU
gpu_list = args.gpus
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
test_logger.info('CUDA_VISIBLE_DEVICES=%s', gpu_list)
test_logger.info('Dataset: %s', dataset_dir)

# LPIPS
import lpips
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips.LPIPS(net='alex').to(device)

def to_lpips_tensor(img_hw3_float01):
    # numpy HxWxC in [0,1] -> torch 1x3xHxW in [-1,1]
    t = torch.from_numpy(img_hw3_float01).permute(2,0,1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0

def rgb01_to_ycbcr01(img_rgb01: np.ndarray):
    """img_rgb01: HxWx3 float32/64 in [0,1] (RGB).
       Returns: y, cb, cr each in [0,1] as float64."""
    ycbcr = rgb2ycbcr(np.clip(img_rgb01, 0.0, 1.0))  # returns [0,255]
    y = ycbcr[..., 0] / 255.0
    cb = ycbcr[..., 1] / 255.0
    cr = ycbcr[..., 2] / 255.0
    return y, cb, cr

def psnr_yc(gt_rgb01, pr_rgb01):
    y_gt, cb_gt, cr_gt = rgb01_to_ycbcr01(gt_rgb01)
    y_pr, cb_pr, cr_pr = rgb01_to_ycbcr01(pr_rgb01)
    psnr_y = metrics.peak_signal_noise_ratio(y_gt, y_pr, data_range=1.0)
    psnr_cb = metrics.peak_signal_noise_ratio(cb_gt, cb_pr, data_range=1.0)
    psnr_cr = metrics.peak_signal_noise_ratio(cr_gt, cr_pr, data_range=1.0)
    return psnr_y, 0.5 * (psnr_cb + psnr_cr)

def ssim_yc(gt_rgb01, pr_rgb01):
    y_gt, cb_gt, cr_gt = rgb01_to_ycbcr01(gt_rgb01)
    y_pr, cb_pr, cr_pr = rgb01_to_ycbcr01(pr_rgb01)
    ssim_y = metrics.structural_similarity(y_gt, y_pr, data_range=1.0)
    ssim_cb = metrics.structural_similarity(cb_gt, cb_pr, data_range=1.0)
    ssim_cr = metrics.structural_similarity(cr_gt, cr_pr, data_range=1.0)
    return ssim_y, 0.5 * (ssim_cb + ssim_cr)


def validate_sorted_pairs(input_paths, target_paths, input_root, target_root):
    """Validate one-to-one relative filenames while retaining sorted pairing.

    Args:
        input_paths (list[str]): Naturally sorted low-light image paths.
        target_paths (list[str]): Naturally sorted ground-truth image paths.
        input_root (str): Root used to derive input relative paths.
        target_root (str): Root used to derive target relative paths.

    Raises:
        ValueError: If counts differ or corresponding relative filenames differ.
    """
    if not input_paths:
        raise ValueError(f'No test images found under {input_root}.')
    if len(input_paths) != len(target_paths):
        raise ValueError(
            f'LQ/GT image counts differ: {len(input_paths)} vs {len(target_paths)}.')

    mismatches = []
    for input_path, target_path in zip(input_paths, target_paths):
        input_key = os.path.normcase(os.path.normpath(
            os.path.relpath(input_path, input_root)))
        target_key = os.path.normcase(os.path.normpath(
            os.path.relpath(target_path, target_root)))
        if input_key != target_key:
            mismatches.append((input_key, target_key))
            if len(mismatches) == 5:
                break
    if mismatches:
        samples = '; '.join(f'{lq} != {gt}' for lq, gt in mismatches)
        raise ValueError(f'LQ/GT filenames are not aligned: {samples}')


def parse_complexity_size(size_text):
    """Parse an HxW complexity input string and validate positive dimensions."""
    try:
        height, width = (int(value) for value in size_text.lower().split('x'))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid --complexity_size {size_text!r}; expected HxW.') from exc
    if height <= 0 or width <= 0:
        raise ValueError('--complexity_size dimensions must be positive.')
    return height, width


def clear_cuda_cache():
    """Release cached CUDA memory when testing on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
# ----------------------------------------------

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False

with open(args.opt, 'r', encoding='utf-8') as f:
    x = yaml.load(f, Loader=Loader)

s = x['network_g'].pop('type')

model_restoration = create_model(opt).net_g
checkpoint = torch.load(args.weights, map_location='cpu')
checkpoint_params = checkpoint.get('params', checkpoint)
checkpoint_params = {
    key[7:] if key.startswith('module.') else key: value
    for key, value in checkpoint_params.items()
}
model_restoration.load_state_dict(checkpoint_params)

test_logger.info('Testing using weights: %s', args.weights)
model_restoration = model_restoration.to(device)
if torch.cuda.is_available():
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Output dirs
factor = 2
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
output_dir = result_dir

# Metrics accumulators
psnr = []
ssim = []
lpips_list = []
psnr_y_list, ssim_y_list = [], []      # <-- NEW
psnr_c_list, ssim_c_list = [], []      # <-- NEW

if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    dopt = opt['datasets']['val']
    dopt['phase'] = 'test'
    if dopt.get('scale') is None:
        dopt['scale'] = 1
    if '~' in dopt['dataroot_gt']:
        dopt['dataroot_gt'] = os.path.expanduser('~') + dopt['dataroot_gt'][1:]
    if '~' in dopt['dataroot_lq']:
        dopt['dataroot_lq'] = os.path.expanduser('~') + dopt['dataroot_lq'][1:]
    dataset_obj = Dataset(dopt)
    print(f'test dataset length: {len(dataset_obj)}')
    dataloader = DataLoader(dataset=dataset_obj, batch_size=1, shuffle=False)

    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            clear_cuda_cache()

            input_ = data_batch['lq'].to(device)
            input_save = data_batch['lq'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # pad
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = self_ensemble(input_, model_restoration) if args.self_ensemble else model_restoration(input_)
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored + 1e-12), 0, 1)

            # RGB overall (keeping your existing)
            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            py, pc = psnr_yc(target, restored)
            sy, sc = ssim_yc(target, restored)
            psnr_y_list.append(py); psnr_c_list.append(pc)
            ssim_y_list.append(sy); ssim_c_list.append(sc)

            # save
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img(os.path.join(result_dir, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'), img_as_ubyte(restored))
            utils.save_img(os.path.join(result_dir_input, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'), img_as_ubyte(input_save))
            utils.save_img(os.path.join(result_dir_gt, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'), img_as_ubyte(target))

else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    test_logger.info('LQ directory: %s', input_dir)
    test_logger.info('GT directory: %s', target_dir)

    input_paths = natsorted(glob(os.path.join(input_dir, '*.png')) +
                            glob(os.path.join(input_dir, '*.jpg')) +
                            glob(os.path.join(input_dir, '*.jpeg')))
    target_paths = natsorted(glob(os.path.join(target_dir, '*.png')) +
                             glob(os.path.join(target_dir, '*.jpg')) +
                             glob(os.path.join(target_dir, '*.jpeg')))
    validate_sorted_pairs(input_paths, target_paths, input_dir, target_dir)
    test_logger.info('Validated %d aligned LQ/GT pairs.', len(input_paths))

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            clear_cuda_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img_t = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img_t.unsqueeze(0).to(device)

            # pad
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 2200 and w < 3050:
                restored = self_ensemble(input_, model_restoration) if args.self_ensemble else model_restoration(input_)
            else:
                # four-way split
                input_1 = input_[:, :, :, 0::4]
                input_2 = input_[:, :, :, 1::4]
                input_3 = input_[:, :, :, 2::4]
                input_4 = input_[:, :, :, 3::4]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                    restored_3 = self_ensemble(input_3, model_restoration)
                    restored_4 = self_ensemble(input_4, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                    restored_3 = model_restoration(input_3)
                    restored_4 = model_restoration(input_4)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 0::4] = restored_1
                restored[:, :, :, 1::4] = restored_2
                restored[:, :, :, 2::4] = restored_3
                restored[:, :, :, 3::4] = restored_4

            # unpad
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(restored, target, args.GT_mean))
            ssim.append(utils.calculate_ssim(restored, target, border=0, gtmean=args.GT_mean))

            py, pc = psnr_yc(target, restored)
            sy, sc = ssim_yc(target, restored)
            psnr_y_list.append(py); psnr_c_list.append(pc)
            ssim_y_list.append(sy); ssim_c_list.append(sc)

            # save image
            save_base = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'
            utils.save_img(os.path.join(output_dir, save_base), img_as_ubyte(restored))

            # LPIPS
            lp_r = to_lpips_tensor(restored.astype(np.float32))
            lp_t = to_lpips_tensor(target.astype(np.float32))
            with torch.no_grad():
                lp_val = lpips_fn(lp_r, lp_t).item()
            lpips_list.append(lp_val)

# Report
psnr_mean = float(np.mean(psnr)) if psnr else float('nan')
ssim_mean = float(np.mean(ssim)) if ssim else float('nan')
lpips_mean = float(np.mean(lpips_list)) if lpips_list else float('nan')

psnr_y_mean = float(np.mean(psnr_y_list)) if psnr_y_list else float('nan')
ssim_y_mean = float(np.mean(ssim_y_list)) if ssim_y_list else float('nan')
psnr_c_mean = float(np.mean(psnr_c_list)) if psnr_c_list else float('nan')
ssim_c_mean = float(np.mean(ssim_c_list)) if ssim_c_list else float('nan')

complexity_height, complexity_width = parse_complexity_size(args.complexity_size)
bare_model = (model_restoration.module
              if hasattr(model_restoration, 'module') else model_restoration)
complexity = _compute_complexity(
    bare_model,
    input_res=(3, complexity_height, complexity_width),
    device=device.type)
if 'error' in complexity:
    raise RuntimeError(
        f"Complexity profiling failed: {complexity['error']}")
params_m = _count_params(bare_model) / 1e6
input_size = f'1x3x{complexity_height}x{complexity_width}'

test_logger.info('RGB  - PSNR: %.4f', psnr_mean)
test_logger.info('RGB  - SSIM: %.4f', ssim_mean)
test_logger.info('LPIPS (alex): %.6f', lpips_mean)
test_logger.info('Y    - PSNR_y: %.4f', psnr_y_mean)
test_logger.info('Y    - SSIM_y: %.4f', ssim_y_mean)
test_logger.info('Ch   - PSNR_c: %.4f (avg of Cb & Cr)', psnr_c_mean)
test_logger.info('Ch   - SSIM_c: %.4f (avg of Cb & Cr)', ssim_c_mean)
test_logger.info(
    'Complexity - Params(M): %.6f, GMACs: %.6f, GFLOPs(2x MACs): %.6f, TFLOPs: %.9f, input: %s',
    params_m, complexity['gmacs'], complexity['gflops'],
    complexity['tflops'], input_size)

dataset_splits = {
    'LOL_v1': ('our485', 'eval15'),
    'LOL_v2_real': ('Real_captured/Train', 'Real_captured/Test'),
    'LOL_v2_synthetic': ('Synthetic/Train', 'Synthetic/Test'),
}
train_split, test_split = dataset_splits.get(dataset, ('', ''))
metric_path = os.path.join(dataset_root, 'metric.csv')
metric_row = {
    'dataset': dataset_dir,
    'train_split': train_split,
    'test_split': test_split,
    'config': config,
    'checkpoint_name': checkpoint_name,
    'psnr': f'{psnr_mean:.6f}',
    'ssim': f'{ssim_mean:.6f}',
    'ssim_mode': 'RGB channel mean',
    'lpips': f'{lpips_mean:.6f}',
    'lpips_backbone': 'alex',
    'lpips_input_range': '[-1,1]',
    'psnr_y': f'{psnr_y_mean:.6f}',
    'ssim_y': f'{ssim_y_mean:.6f}',
    'psnr_chroma': f'{psnr_c_mean:.6f}',
    'ssim_chroma': f'{ssim_c_mean:.6f}',
    'params_m': f'{params_m:.6f}',
    'gmacs': f"{complexity['gmacs']:.6f}",
    'gflops': f"{complexity['gflops']:.6f}",
    'tflops': f"{complexity['tflops']:.9f}",
    'input_size': input_size,
    'checkpoint': os.path.normpath(args.weights).replace('\\', '/'),
    'enhanced_images': os.path.normpath(result_dir).replace('\\', '/'),
    'metric_source': 'Enhancement/test.py',
    'complexity_tool': 'thop.profile',
    'complexity_note': 'THOP returns MACs; GFLOPs=2*MACs/1e9; TFLOPs=2*MACs/1e12',
}
with open(metric_path, 'w', newline='', encoding='utf-8-sig') as metric_file:
    writer = csv.DictWriter(metric_file, fieldnames=metric_row.keys())
    writer.writeheader()
    writer.writerow(metric_row)
test_logger.info('Enhanced images: %s', result_dir)
test_logger.info('Metrics CSV: %s', metric_path)
