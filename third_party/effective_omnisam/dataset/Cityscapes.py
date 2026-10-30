import os
import random
from PIL import Image

import torch
import numpy as np
from torch.utils import data
from torchvision import transforms


def _normalize_cityscapes_rel_name(name, split):
    name = name.strip().replace("\\", "/")
    if "leftImg8bit/" in name:
        name = name.split("leftImg8bit/", 1)[1]
    if name.startswith(split + "/"):
        name = name[len(split) + 1:]
    if "/" not in name:
        city = name.split("_")[0]
        name = city + "/" + name
    return name


def _build_cityscapes_files(root, list_path, split):
    files = []
    missing = []
    img_ids = [i_id.strip() for i_id in open(list_path)]
    for name in img_ids:
        rel_name = _normalize_cityscapes_rel_name(name, split)
        img_file = os.path.join(root, f"leftImg8bit/{split}/{rel_name}")
        img_base = os.path.basename(rel_name)
        lbname = img_base.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        rel_dir = os.path.dirname(rel_name)
        label_rel = os.path.join(rel_dir, lbname) if rel_dir else lbname
        label_file = os.path.join(root, f"gtFine/{split}/{label_rel}")

        if os.path.isfile(img_file) and os.path.isfile(label_file):
            files.append({
                "img": img_file,
                "label": label_file,
                "name": img_base,
            })
        else:
            missing.append((img_file, label_file))

    if not files:
        preview = missing[:3]
        raise FileNotFoundError(
            f"No valid Cityscapes samples found for split={split}, list={list_path}. "
            f"Examples: {preview}"
        )

    if missing:
        print(
            f"[Cityscapes13] split={split}: kept {len(files)}/{len(img_ids)} samples, "
            f"skipped {len(missing)} missing entries.",
            flush=True,
        )

    return files


class Cityscapes13(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.stride, self.window_size = sliding_window
        self.files = _build_cityscapes_files(self.root, self.list_path, self.set)
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)

    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])
        if np.random.randint(0, 2):
            patched_image.reverse()
            patched_label.reverse()
        return torch.stack(patched_image), torch.stack(patched_label)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)

        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        return image, label, np.array(size), name


class Cityscapes19(data.Dataset):
    """
    Cityscapes dataset in 19-class TrainIds space (0..18, 255 ignore).
    """

    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.stride, self.window_size = sliding_window
        self.files = _build_cityscapes_files(self.root, self.list_path, self.set)

    def __len__(self):
        return len(self.files)

    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])
        if np.random.randint(0, 2):
            patched_image.reverse()
            patched_label.reverse()
        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        return image, label, np.array(size), name


class Cityscapes13forVal(data.Dataset):
    def __init__(self, root, list_path, crop_size=(321, 321), sliding_window=None, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.stride, self.window_size = sliding_window
        self.files = _build_cityscapes_files(self.root, self.list_path, self.set)
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)
        
    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])

        patched_image.extend(reversed(patched_image))
        patched_label.extend(reversed(patched_label))
        return torch.stack(patched_image), torch.stack(patched_label)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        ori_label = label
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        ori_label = torch.LongTensor(np.array(ori_label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)

        return image, label, np.array(size), name, ori_label
