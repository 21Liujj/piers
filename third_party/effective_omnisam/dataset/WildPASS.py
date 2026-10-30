import os
import random
from PIL import Image

import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from utils.depth_prior import find_depth_path, load_depth, DEFAULT_DEPTH_EXTS
from utils.wildpass_label_convert import convert_wildpass_raw_to_train13

class WildPASSforPL(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        crop_size=(2048, 400),
        sliding_window=None,
        depth_prior_root: str = "",
        depth_file_exts=None,
        depth_normalize: str = "minmax",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.files = []
        self.depth_prior_root = depth_prior_root or ""
        self.depth_file_exts = tuple(depth_file_exts) if depth_file_exts else DEFAULT_DEPTH_EXTS
        self.depth_normalize = depth_normalize
        self._depth_missing_warned = False
        self.depth_loading_missing_count = 0

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.target_source = ['Amsterdam', 'Athens', 'Auckland', 'Bangkok', 'Barcelona', 'Bremen', 'Brussel', 
                            'Buenosaires', 'Canberra', 'Capetown', 'Chicago', 'Copenhagen', 'Dublin', 'Edinburgh',
                            'Florence', 'Glasgow', 'Helsinki', 'Hochiminhcity', 'Istanbul', 'Jakarta', 'Lisbon', 
                            'Manila', 'Marseille', 'Melbourne', 'Mexicocity', 'Montreal', 'Moscow', 'Nottingham', 
                            'Osaka', 'Oslo', 'Sandiego', 'Saopaulo', 'Seoul', 'Singapore', 'Stockholm', 'Toronto',
                            'Turin', 'Yokohama', 'Zagreb', 'Zurich']

        for target_name in self.target_source:
            for name in self.img_ids:
                if target_name in name:
                    img_file = os.path.join(self.root, "leftImg8bit/%s" % (name))
                    self.files.append({
                        "img": img_file,
                        "name": name
                    })

    def __len__(self):
        return len(self.files)

    def _sliding_windows(self, image, window_size, stride):
        _, h, w = image.shape
        patched_image = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
        patched_image.extend(reversed(patched_image))
        return torch.stack(patched_image)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        image = self._sliding_windows(image, window_size=self.window_size, stride=self.stride)

        depth_patch = None
        if self.depth_prior_root:
            try:
                prior_path = find_depth_path(datafiles["img"], self.depth_prior_root, self.depth_file_exts)
                if prior_path:
                    depth_1hw = load_depth(
                        prior_path,
                        target_size=(self.crop_size[1], self.crop_size[0]),
                        normalize=self.depth_normalize,
                    )
                    depth_hw = depth_1hw.squeeze(0)
                    depth_patch = self._sliding_windows(depth_hw.unsqueeze(0), window_size=self.window_size, stride=self.stride).float()
                else:
                    self.depth_loading_missing_count += 1
                    if not self._depth_missing_warned:
                        print(
                            f"[WildPASSforPL] pseudo-depth missing under '{self.depth_prior_root}', fallback to no-depth in PL.",
                            flush=True,
                        )
                        self._depth_missing_warned = True
            except Exception:
                depth_patch = None

        if depth_patch is None:
            return image, size, name
        return image, size, name, depth_patch

class WildPASSforAdaptation(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        crop_size=(2048, 400),
        sliding_window=None,
        data_root="data",
        depth_prior_root: str = "",
        depth_file_exts=None,
        depth_normalize: str = "minmax",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.files = []
        self.data_root = data_root
        self.depth_prior_root = depth_prior_root or ""
        self.depth_file_exts = tuple(depth_file_exts) if depth_file_exts else DEFAULT_DEPTH_EXTS
        self.depth_normalize = depth_normalize
        self._depth_missing_warned = False
        self.depth_loading_missing_count = 0

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        missing = []
        for name in self.img_ids:
            rel_name = name.lstrip("/")
            img_file = os.path.join(self.data_root, "WildPASS", f"leftImg8bit/{rel_name}")
            lbname = rel_name.replace(".jpg", "_labelTrainIds.png")
            label_file = os.path.join(self.root, f"gtFine/{lbname}")
            if os.path.isfile(img_file) and os.path.isfile(label_file):
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": rel_name,
                })
            else:
                missing.append((img_file, label_file))
        if not self.files:
            raise FileNotFoundError(
                f"No valid WildPASS adaptation samples found from {list_path}. Examples: {missing[:3]}"
            )
        if missing:
            print(
                f"[WildPASSforAdaptation] kept {len(self.files)}/{len(self.img_ids)} samples, skipped {len(missing)} missing entries.",
                flush=True,
            )
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        return new_mask
    
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

        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        size = np.asarray(label).shape 

        depth_patch = None
        if self.depth_prior_root:
            try:
                prior_path = find_depth_path(datafiles["img"], self.depth_prior_root, self.depth_file_exts)
                if prior_path:
                    depth_1hw = load_depth(
                        prior_path,
                        target_size=(self.crop_size[1], self.crop_size[0]),
                        normalize=self.depth_normalize,
                    )
                    depth_hw = depth_1hw.squeeze(0)
                    _, depth_p = self._sliding_windows(image, depth_hw, window_size=self.window_size, stride=self.stride)
                    depth_patch = depth_p.unsqueeze(1).to(dtype=torch.float32)
                else:
                    self.depth_loading_missing_count += 1
                    if not self._depth_missing_warned:
                        print(
                            f"[WildPASSforAdaptation] pseudo-depth missing under '{self.depth_prior_root}', fallback to no-depth.",
                            flush=True,
                        )
                        self._depth_missing_warned = True
            except Exception:
                depth_patch = None

        if depth_patch is None:
            return image, label, size, name
        return image, label, size, name, depth_patch


class WildPASS13(data.Dataset):
    """
    WildPASS 带语义标签的验证集（如 leftImg8bit/val/cs 与 gtFine/val/cs 同名 png）。
    滑窗与双 pass 逻辑与 DensePASS13 一致，默认全景尺寸 2048×400。

    layout="flat": root/{image_dir}/{name} 与 root/{label_dir}/{name}（lists 中为相对文件名，如 1.png）。
    """

    def __init__(
        self,
        root,
        list_path,
        crop_size=(2048, 400),
        sliding_window=None,
        passes=2,
        set_name="val",
        layout="cityscapes",
        image_dir="images",
        label_dir="labels_trainid",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set_name = set_name
        self.layout = layout
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.stride, self.window_size = sliding_window
        self.files = []
        assert passes in [1, 2]
        self.passes = passes

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12])

        for name in self.img_ids:
            rel = name.lstrip("/")
            if layout == "flat":
                img_file = os.path.join(self.root, self.image_dir, rel)
                lab_name = rel
                if self.label_dir and not os.path.isfile(os.path.join(self.root, self.label_dir, lab_name)):
                    stem, _ext = os.path.splitext(rel)
                    lab_name = stem + ".png"
                label_file = os.path.join(self.root, self.label_dir, lab_name)
                self.files.append({"img": img_file, "label": label_file, "name": rel})
            else:
                img_file = os.path.join(self.root, "leftImg8bit", self.set_name, rel)
                label_file = os.path.join(self.root, "gtFine", self.set_name, rel)
                self.files.append({"img": img_file, "label": label_file, "name": rel})

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255:
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        return new_mask

    @staticmethod
    def _cityscapes_label_id_to_train_id(mask: np.ndarray) -> np.ndarray:
        """Cityscapes labelId（0–33）→ trainId（0–18, 255），与 torchvision.datasets.Cityscapes 一致。"""
        from torchvision.datasets import Cityscapes

        lut = np.zeros(256, dtype=np.int32)
        lut[:] = 255
        for c in Cityscapes.classes:
            if c.id is not None and c.id >= 0:
                tid = c.train_id
                lut[c.id] = 255 if tid in (-1, 255) else int(tid)
        return lut[mask.astype(np.int64)]

    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x : x + window_size])
                patched_label.append(label[:, x : x + window_size])
        if self.passes == 2:
            patched_image.extend(reversed(patched_image))
            patched_label.extend(reversed(patched_label))
        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["label"])

        raw = np.array(label).astype("int32")
        mapped, _enc = convert_wildpass_raw_to_train13(raw)
        label = Image.fromarray(mapped.astype(np.uint8))
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        # 与推理全景一致；勿在 resize 前缓存 ori_label（否则与 merge 后 pred 的网格不对齐）
        ori_label = torch.LongTensor(np.asarray(label, dtype=np.int32))

        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        label = torch.LongTensor(np.asarray(label, dtype=np.int32))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        size = np.asarray(label).shape

        return image, label, size, name, ori_label


class WildPASS19(data.Dataset):
    """
    WildPASS validation set in 19-class TrainIds space (0..18, 255 ignore).
    Sliding window and double-pass logic match WildPASS13, but without 19->13 mapping.
    """

    def __init__(
        self,
        root,
        list_path,
        crop_size=(2048, 400),
        sliding_window=None,
        passes=2,
        set_name="val",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set_name = set_name
        self.stride, self.window_size = sliding_window
        self.files = []
        assert passes in [1, 2]
        self.passes = passes

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self._allowed_train = set(range(19)) | {255}

        for name in self.img_ids:
            rel = name.lstrip("/")
            img_file = os.path.join(self.root, "leftImg8bit", self.set_name, rel)
            label_file = os.path.join(self.root, "gtFine", self.set_name, rel)
            self.files.append({"img": img_file, "label": label_file, "name": rel})

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _cityscapes_label_id_to_train_id(mask: np.ndarray) -> np.ndarray:
        from torchvision.datasets import Cityscapes

        lut = np.zeros(256, dtype=np.int32)
        lut[:] = 255
        for c in Cityscapes.classes:
            if c.id is not None and c.id >= 0:
                tid = c.train_id
                lut[c.id] = 255 if tid in (-1, 255) else int(tid)
        return lut[mask.astype(np.int64)]

    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x : x + window_size])
                patched_label.append(label[:, x : x + window_size])
        if self.passes == 2:
            patched_image.extend(reversed(patched_image))
            patched_label.extend(reversed(patched_label))
        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["label"])

        raw = np.array(label).astype("int32")
        if not set(np.unique(raw).tolist()).issubset(self._allowed_train):
            raw = self._cityscapes_label_id_to_train_id(raw)
        label = Image.fromarray(raw)
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        ori_label = torch.LongTensor(np.asarray(label, dtype=np.int32))

        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        label = torch.LongTensor(np.asarray(label, dtype=np.int32))
        image, label = self._sliding_windows(
            image, label, window_size=self.window_size, stride=self.stride
        )
        size = np.asarray(label).shape

        return image, label, size, name, ori_label


class WildPASSforAdaptation19(data.Dataset):
    """
    WildPASS pseudo-label dataset for adaptation in 19-class TrainIds space.
    Expects gtFine/*_labelTrainIds.png produced by pseudo-label generator.
    """

    def __init__(
        self,
        root,
        list_path,
        crop_size=(2048, 400),
        sliding_window=None,
        data_root="data",
        depth_prior_root: str = "",
        depth_file_exts=None,
        depth_normalize: str = "minmax",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.files = []
        self.data_root = data_root
        self.depth_prior_root = depth_prior_root or ""
        self.depth_file_exts = tuple(depth_file_exts) if depth_file_exts else DEFAULT_DEPTH_EXTS
        self.depth_normalize = depth_normalize
        self._depth_missing_warned = False
        self.depth_loading_missing_count = 0

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        missing = []
        for name in self.img_ids:
            rel_name = name.lstrip("/")
            img_file = os.path.join(self.data_root, "WildPASS", f"leftImg8bit/{rel_name}")
            lbname = rel_name.replace(".jpg", "_labelTrainIds.png")
            label_file = os.path.join(self.root, f"gtFine/{lbname}")
            if os.path.isfile(img_file) and os.path.isfile(label_file):
                self.files.append(
                    {
                        "img": img_file,
                        "label": label_file,
                        "name": rel_name,
                    }
                )
            else:
                missing.append((img_file, label_file))
        if not self.files:
            raise FileNotFoundError(
                f"No valid WildPASS19 adaptation samples found from {list_path}. Examples: {missing[:3]}"
            )
        if missing:
            print(
                f"[WildPASSforAdaptation19] kept {len(self.files)}/{len(self.img_ids)} samples, skipped {len(missing)} missing entries.",
                flush=True,
            )

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
        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype("int32"))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        size = np.asarray(label).shape
        depth_patch = None
        if self.depth_prior_root:
            try:
                prior_path = find_depth_path(datafiles["img"], self.depth_prior_root, self.depth_file_exts)
                if prior_path:
                    depth_1hw = load_depth(
                        prior_path,
                        target_size=(self.crop_size[1], self.crop_size[0]),
                        normalize=self.depth_normalize,
                    )
                    depth_hw = depth_1hw.squeeze(0)
                    _, depth_p = self._sliding_windows(image, depth_hw, window_size=self.window_size, stride=self.stride)
                    depth_patch = depth_p.unsqueeze(1).to(dtype=torch.float32)
                else:
                    self.depth_loading_missing_count += 1
                    if not self._depth_missing_warned:
                        print(
                            f"[WildPASSforAdaptation19] pseudo-depth missing under '{self.depth_prior_root}', fallback to no-depth.",
                            flush=True,
                        )
                        self._depth_missing_warned = True
            except Exception:
                depth_patch = None

        if depth_patch is None:
            return image, label, size, name
        return image, label, size, name, depth_patch