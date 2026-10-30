"""
Circular Padding Versions of DensePASS Dataset for Seamless Circular Memory (SCM)

Author: Research Team
Date: 2026-08-28
Branch: feature/scm-circular-memory

Key modification:
- Add circular padding on width dimension after resize and before sliding window
- Padding width = window_size // 2 to ensure boundary windows can access opposite side
- Only applied during validation (training keeps original)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms

from dataset.DensePASS import DensePASS13, DensePASS19


class DensePASS13Circular(DensePASS13):
    """
    DensePASS13 with circular padding support for 360° panorama seamless inference.
    
    Pipeline:
    1. Load and resize image to (4096, 1024) - same as parent
    2. Apply circular padding on width: (4096, 1024) -> (4096+2*pad, 1024)
    3. Sliding window inference on padded image
    4. Remove padding from final prediction
    """
    
    def __init__(self, root, list_path, crop_size=(4096, 1024),
                 sliding_window=None, passes=1, set='val',
                 enable_circular_padding=True,
                 circular_pad_width=512):
        """
        Args:
            enable_circular_padding: bool, enable circular padding
            circular_pad_width: int, padding width in pixels (default: window_size//2 = 512)
        """
        super().__init__(root, list_path, crop_size, sliding_window, passes, set)
        
        self.enable_circular_padding = enable_circular_padding
        self.circular_pad_width = circular_pad_width
        
        if self.enable_circular_padding:
            print(f"[DensePASS13Circular] Initialized with:")
            print(f"  - crop_size: {crop_size}")
            print(f"  - circular_padding: ENABLED")
            print(f"  - pad_width: {circular_pad_width}px")
            print(f"  - padded_width: {crop_size[0] + 2 * circular_pad_width}px")

    def _get_frame_cnt(self):
        """
        Override to return correct frame count when circular padding is enabled.
        """
        w, h = self.crop_size
        if self.enable_circular_padding:
            w_padded = w + 2 * self.circular_pad_width
            frame_cnt = (w_padded - self.window_size) // self.stride + 1
        else:
            frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt
    
    def _sliding_windows(self, image, label, window_size, stride):
        """
        Override parent's sliding window to support circular padded input.
        
        If circular padding is enabled, image/label width > crop_size[0].
        We process all patches including those that span the original boundary.
        """
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
    
    def __getitem__(self, index):
        """
        Main data loading with circular padding.
        """
        datafiles = self.files[index]
        
        # Load images
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        
        # Map 19->13 classes
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]
        
        # Resize to crop_size
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        ori_label = torch.LongTensor(np.asarray(label, dtype=np.int32))
        
        # To tensor
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)  # [C, H, W]
        label = torch.LongTensor(np.asarray(label, dtype=np.int32))  # [H, W]
        
        # Apply circular padding (only for validation)
        if self.enable_circular_padding and self.set == 'val':
            # Pad on width dimension: (left, right, top, bottom)
            image = F.pad(
                image, 
                pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                mode='circular'
            )
            
            # Label padding (need to convert to float for F.pad, then back to long)
            label_4d = label.unsqueeze(0).unsqueeze(0).float()
            label_padded = F.pad(
                label_4d,
                pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                mode='circular'
            )
            label = label_padded.squeeze(0).squeeze(0).long()
        
        # Sliding window (now works on padded image if enabled)
        image, label = self._sliding_windows(image, label, 
                                            window_size=self.window_size, 
                                            stride=self.stride)
        size = np.asarray(label).shape
        
        return image, label, size, name, ori_label


class DensePASS19Circular(DensePASS19):
    """
    DensePASS19 (19-class) with circular padding support.
    """
    
    def __init__(self, root, list_path, crop_size=(4096, 1024),
                 sliding_window=None, passes=1, set='val',
                 enable_circular_padding=True,
                 circular_pad_width=512):
        super().__init__(root, list_path, crop_size, sliding_window, passes, set)
        
        self.enable_circular_padding = enable_circular_padding
        self.circular_pad_width = circular_pad_width
        
        if self.enable_circular_padding:
            print(f"[DensePASS19Circular] Initialized with:")
            print(f"  - crop_size: {crop_size}")
            print(f"  - circular_padding: ENABLED")
            print(f"  - pad_width: {circular_pad_width}px")

    def _get_frame_cnt(self):
        w, h = self.crop_size
        if self.enable_circular_padding:
            w_padded = w + 2 * self.circular_pad_width
            frame_cnt = (w_padded - self.window_size) // self.stride + 1
        else:
            frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt
    
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
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        
        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["label"])
        
        label_np = np.array(label).astype("int32")
        label = Image.fromarray(label_np)
        name = datafiles["name"]
        
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        ori_label = torch.LongTensor(np.asarray(label, dtype=np.int32))
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.asarray(label, dtype=np.int32))
        
        # Circular padding
        if self.enable_circular_padding and self.set == 'val':
            image = F.pad(
                image, 
                pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                mode='circular'
            )
            
            label_4d = label.unsqueeze(0).unsqueeze(0).float()
            label_padded = F.pad(
                label_4d,
                pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                mode='circular'
            )
            label = label_padded.squeeze(0).squeeze(0).long()
        
        image, label = self._sliding_windows(
            image, label, window_size=self.window_size, stride=self.stride
        )
        size = np.asarray(label).shape
        
        return image, label, size, name, ori_label
