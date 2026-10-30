"""
Flat-layout Stanford360 / SPan8 panoramas (lists + images/ + labels_trainid/).
Aligns with StanfordPan8* sliding-window API for Engine and validation.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


def _read_list(list_path: str) -> List[str]:
    p = Path(list_path)
    lines: List[str] = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(Path(line).name)
    return lines


class Stanford360PanoramaVal(data.Dataset):
    """
    Validation: grayscale trainId labels in label_dir (0-7, 255), no color2id.
    Optional legacy crop (0,320,w,1728) like StanfordPan8forVal.
    """

    def __init__(
        self,
        root: str,
        list_path: str,
        crop_size: Tuple[int, int] = (3072, 1024),
        sliding_window: Tuple[int, int] = (256, 1024),
        passes: int = 2,
        image_dir: str = "images",
        label_dir: str = "labels_trainid",
        legacy_area_crop: bool = False,
    ):
        super().__init__()
        self.root = root
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.passes = passes
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.legacy_area_crop = legacy_area_crop
        self.names = _read_list(list_path)
        self.files = []
        r = Path(root)
        for name in self.names:
            ip = r / image_dir / name
            lp = r / label_dir / name
            if not lp.is_file():
                stem = Path(name).stem
                found = list((r / label_dir).glob(stem + ".*"))
                if found:
                    lp = found[0]
            self.files.append({"img": str(ip), "lbl": str(lp), "name": name})

    def __len__(self) -> int:
        return len(self.files)

    def _sliding_windows(self, image: torch.Tensor, label: torch.Tensor, window_size: int, stride: int):
        patched_image = []
        patched_label = []
        if label.dim() == 2:
            _h, w = label.shape
        else:
            _, _h, w = label.shape
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x : x + window_size])
                patched_label.append(label[:, x : x + window_size])
        if self.passes == 2:
            patched_image.extend(reversed(patched_image))
            patched_label.extend(reversed(patched_label))
        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self) -> int:
        w, _h = self.crop_size
        return (w - self.window_size) // self.stride + 1

    def __getitem__(self, index: int):
        rec = self.files[index]
        image = Image.open(rec["img"]).convert("RGB")
        label = Image.open(rec["lbl"])
        if label.mode != "L":
            label = label.convert("L")
        width, height = image.size
        if self.legacy_area_crop:
            left, top, right, bottom = 0, 320, width, min(1728, height)
            image = image.crop((left, top, right, bottom))
            label = label.crop((left, top, right, bottom))

        label_np = np.array(label).astype(np.int32)
        label = Image.fromarray(label_np.astype(np.uint8))

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        ori_label = label.copy()

        size = np.asarray(image).shape
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        label_t = torch.LongTensor(np.array(label).astype(np.int32))
        ori_t = torch.LongTensor(np.array(ori_label).astype(np.int32))
        image, label_t = self._sliding_windows(image, label_t, self.window_size, self.stride)
        return image, label_t, np.array(size), rec["name"], ori_t


class Stanford360PanoramaPL(data.Dataset):
    """Pseudo-label generation: images only (matches StanfordPan8forPL)."""

    def __init__(
        self,
        root: str,
        list_path: str,
        crop_size: Tuple[int, int] = (3072, 1024),
        sliding_window: Tuple[int, int] = (256, 1024),
        image_dir: str = "images",
        legacy_area_crop: bool = False,
    ):
        super().__init__()
        self.root = root
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.image_dir = image_dir
        self.legacy_area_crop = legacy_area_crop
        self.names = _read_list(list_path)
        self.files = [str(Path(root) / image_dir / n) for n in self.names]

    def __len__(self) -> int:
        return len(self.files)

    def _sliding_windows(self, image: torch.Tensor, window_size: int, stride: int):
        _, _h, w = image.shape
        patched = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched.append(image[:, :, x : x + window_size])
        patched.extend(reversed(patched))
        return torch.stack(patched)

    def _get_frame_cnt(self) -> int:
        w, _h = self.crop_size
        return (w - self.window_size) // self.stride + 1

    def __getitem__(self, index: int):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        if self.legacy_area_crop:
            left, top, right, bottom = 0, 320, width, min(1728, height)
            image = image.crop((left, top, right, bottom))
        name = Path(path).name
        image = image.resize(self.crop_size, Image.BICUBIC)
        size = np.asarray(image).shape
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        image = self._sliding_windows(image, self.window_size, self.stride)
        return image, np.array(size), name


class Stanford360FlatAdaptation(data.Dataset):
    """
    Adaptation target: pseudo labels under pd_root (flat png trainIds), images under image_root/images/.
    """

    def __init__(
        self,
        pd_root: str,
        used_list_path: str,
        image_root: str,
        crop_size: Tuple[int, int] = (3072, 1024),
        sliding_window: Tuple[int, int] = (256, 1024),
        image_dir: str = "images",
        legacy_area_crop: bool = False,
    ):
        super().__init__()
        self.pd_root = Path(pd_root)
        self.image_root = Path(image_root)
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.legacy_area_crop = legacy_area_crop
        names = []
        with open(used_list_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(Path(line).name)
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def _sliding_windows(self, image: torch.Tensor, label: torch.Tensor, window_size: int, stride: int):
        patched_image = []
        patched_label = []
        if label.dim() == 2:
            _h, w = label.shape
        else:
            _, _h, w = label.shape
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x : x + window_size])
                patched_label.append(label[:, x : x + window_size])
        if np.random.randint(0, 2):
            patched_image.reverse()
            patched_label.reverse()
        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self) -> int:
        w, _h = self.crop_size
        return (w - self.window_size) // self.stride + 1

    def _find_label_path(self, basename: str) -> Path:
        stem = Path(basename).stem
        for p in self.pd_root.rglob(basename):
            if p.is_file():
                return p
        for p in self.pd_root.rglob(stem + ".*"):
            if p.is_file() and p.suffix.lower() in (".png", ".tif", ".jpg"):
                return p
        return self.pd_root / basename

    def __getitem__(self, index: int):
        name = self.names[index]
        ip = self.image_root / self.image_dir / name
        if not ip.is_file():
            stem = Path(name).stem
            for ext in (".png", ".jpg", ".jpeg"):
                alt = self.image_root / self.image_dir / (stem + ext)
                if alt.is_file():
                    ip = alt
                    break
        image = Image.open(ip).convert("RGB")
        lp = self._find_label_path(name)
        label = Image.open(lp)
        if label.mode != "L":
            label = label.convert("L")
        width, height = image.size
        if self.legacy_area_crop:
            left, top, right, bottom = 0, 320, width, min(1728, height)
            image = image.crop((left, top, right, bottom))
            label = label.crop((left, top, right, bottom))

        label_np = np.array(label).astype(np.int32)
        label = Image.fromarray(np.clip(label_np, 0, 255).astype(np.uint8))

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        size = np.asarray(image).shape
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        label_t = torch.LongTensor(np.array(label).astype(np.int32))
        image, label_t = self._sliding_windows(image, label_t, self.window_size, self.stride)
        return image, label_t, np.array(size), name
