import json
import os
import random
from collections import defaultdict

import cv2
import monai.transforms as mt
import numpy as np
import torch
import yaml


def split_dataset(clinical_path, ratio=0.8):
    with open(clinical_path, "r", encoding="utf-8") as f:
        clinical_info = json.load(f)

    dic = defaultdict(list)
    for info in clinical_info:
        dic[info["label"]].append(info)

    train_info = []
    val_info = []

    for label, data in dic.items():
        random.shuffle(data)
        num_samples = len(data)
        train_num = int(ratio * num_samples)
        train_info.extend(data[:train_num])
        val_info.extend(data[train_num:])

    return train_info, val_info


class MyDataset(torch.utils.data.Dataset):
    """
    AutoDL no-padding dataloader.

    Compared with the original dataloader, this keeps the same use_seg image
    loading, MG mask crop, MG left/right normalization, MG CLAHE, and US ROI
    crop, but replaces keep-aspect-ratio resize + zero padding with direct
    stretch resize to config img_rows x img_cols. This removes artificial black
    borders from model inputs at the cost of tolerating aspect-ratio distortion.
    """

    def __init__(self, infos, config_path, use_seg=False, is_train=False):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        data_dir = config["data_dir"]
        img_MG = config["imgMG_dir"]
        mask_MG = config["maskMG_dir"]
        img_US = config["imgUS_dir"]
        mask_US = config["maskUS_dir"]

        self.clinical = {info["id"]: info for info in infos}
        self.img_MG_dir = os.path.join(data_dir, img_MG)
        self.mask_MG_dir = os.path.join(data_dir, mask_MG)
        self.img_US_dir = os.path.join(data_dir, img_US)
        self.mask_US_dir = os.path.join(data_dir, mask_US)
        self.ids = [info["id"] for info in infos]
        self.use_seg = use_seg
        self.is_train = is_train
        self.age_mean = config["age_mean"]
        self.age_std = config["age_std"]
        self.diameter_mean = config["diameter_mean"]
        self.diameter_std = config["diameter_std"]
        self.size = (config["img_rows"], config["img_cols"])

        self.transform = mt.Compose(
            [
                mt.LoadImage(image_only=True),
                mt.EnsureChannelFirst(channel_dim=-1),
                mt.ScaleIntensity(),
                mt.ToTensor(),
            ]
        )

        self.mask_transform = mt.Compose(
            [
                mt.LoadImage(image_only=True),
                mt.EnsureChannelFirst(channel_dim=-1),
                mt.ToTensor(),
            ]
        )

    def _resize_image_direct(self, image):
        target_h, target_w = int(self.size[0]), int(self.size[1])
        image = np.ascontiguousarray(image, dtype=np.float32)

        if image.size == 0:
            image = np.zeros((target_h, target_w), dtype=np.float32)
        else:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    def _resize_direct(self, image, mask):
        target_h, target_w = int(self.size[0]), int(self.size[1])
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)

        if image.size == 0 or mask.size == 0:
            image = np.zeros((target_h, target_w), dtype=np.float32)
            mask = np.zeros((target_h, target_w), dtype=np.float32)
        else:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0.5).astype(np.float32)
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def process_mg_original(self, img_tensor):
        img_np = img_tensor[0].numpy()

        _, w = img_np.shape
        if w > 1:
            left_sum = np.sum(img_np[:, : w // 2])
            right_sum = np.sum(img_np[:, w // 2 :])

            if right_sum > left_sum:
                img_np = np.fliplr(img_np)

        img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_uint8 = np.ascontiguousarray(img_uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_uint8)
        img_np = img_eq.astype(np.float32) / 255.0

        return self._resize_image_direct(img_np)

    def process_us_original(self, img_tensor):
        img_np = img_tensor[0].numpy()
        return self._resize_image_direct(img_np)

    def process_mg(self, img_tensor, mask_tensor):
        img_np = img_tensor[0].numpy()
        mask_np = mask_tensor[0].numpy()

        rows, cols = np.where(mask_np > 0)
        if len(rows) > 0:
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            img_crop = img_np[min_r : max_r + 1, min_c : max_c + 1]
            mask_crop = mask_np[min_r : max_r + 1, min_c : max_c + 1]
        else:
            img_crop = img_np
            mask_crop = mask_np

        _, w = img_crop.shape
        if w > 1:
            left_sum = np.sum(img_crop[:, : w // 2])
            right_sum = np.sum(img_crop[:, w // 2 :])

            if right_sum > left_sum:
                img_crop = np.fliplr(img_crop)
                mask_crop = np.fliplr(mask_crop)

        img_uint8 = (np.clip(img_crop, 0, 1) * 255).astype(np.uint8)
        img_uint8 = np.ascontiguousarray(img_uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_uint8)
        img_crop = img_eq.astype(np.float32) / 255.0

        return self._resize_direct(img_crop, mask_crop)

    def process_us(self, img_tensor, mask_tensor):
        img_np = img_tensor[0].numpy()
        mask_np = mask_tensor[0].numpy()
        h_orig, w_orig = img_np.shape

        rows, cols = np.where(mask_np > 0)
        if len(rows) > 0:
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)

            tumor_h = max(max_r - min_r + 1, 1)
            tumor_w = max(max_c - min_c + 1, 1)
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2

            new_h = tumor_h * 1.5
            new_w = tumor_w * 1.5

            roi_min_r = int(max(0, center_r - new_h / 2))
            roi_max_r = int(min(h_orig, center_r + new_h / 2))
            roi_min_c = int(max(0, center_c - new_w / 2))
            roi_max_c = int(min(w_orig, center_c + new_w / 2))

            if roi_max_r <= roi_min_r:
                roi_max_r = min(h_orig, roi_min_r + 1)
            if roi_max_c <= roi_min_c:
                roi_max_c = min(w_orig, roi_min_c + 1)

            img_crop = img_np[roi_min_r:roi_max_r, roi_min_c:roi_max_c]
            mask_crop = mask_np[roi_min_r:roi_max_r, roi_min_c:roi_max_c]
        else:
            img_crop = img_np
            mask_crop = mask_np

        return self._resize_direct(img_crop, mask_crop)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        patient_id = self.ids[index]
        info = self.clinical[patient_id]

        age = info["age"]
        clinical = [
            (age - self.age_mean) / self.age_std,
            float(info["menopausal_state"]),
        ]
        clinical = torch.tensor(clinical, dtype=torch.float32)

        loc = info["tumor_location"]
        img_MG_CC_path = os.path.join(self.img_MG_dir, f"{patient_id}_{loc * 2 + 1}.nii.gz")
        mask_MG_CC_path = os.path.join(self.mask_MG_dir, f"{patient_id}_{loc * 2 + 1}.nii.gz")

        img_MG_MLO_path = os.path.join(self.img_MG_dir, f"{patient_id}_{loc * 2 + 2}.nii.gz")
        mask_MG_MLO_path = os.path.join(self.mask_MG_dir, f"{patient_id}_{loc * 2 + 2}.nii.gz")

        img_US_path = os.path.join(self.img_US_dir, f"{patient_id}_{loc + 1}.nii.gz")
        mask_US_path = os.path.join(self.mask_US_dir, f"{patient_id}_{loc + 1}.nii.gz")

        img_MG_CC = self.transform(img_MG_CC_path)
        img_MG_MLO = self.transform(img_MG_MLO_path)
        img_US = self.transform(img_US_path)

        label = torch.tensor(info["label"], dtype=torch.long)

        if self.use_seg:
            mask_MG_CC = (self.mask_transform(mask_MG_CC_path) > 0.5).float()
            mask_MG_MLO = (self.mask_transform(mask_MG_MLO_path) > 0.5).float()
            mask_US = (self.mask_transform(mask_US_path) > 0.5).float()

            img_MG_CC, mask_MG_CC = self.process_mg(img_MG_CC, mask_MG_CC)
            img_MG_MLO, mask_MG_MLO = self.process_mg(img_MG_MLO, mask_MG_MLO)
            img_US, mask_US = self.process_us(img_US, mask_US)

            return (img_MG_CC, mask_MG_CC), (img_MG_MLO, mask_MG_MLO), (img_US, mask_US), label, clinical

        img_MG_CC = self.process_mg_original(img_MG_CC)
        img_MG_MLO = self.process_mg_original(img_MG_MLO)
        img_US = self.process_us_original(img_US)

        return img_MG_CC, img_MG_MLO, img_US, label, clinical
