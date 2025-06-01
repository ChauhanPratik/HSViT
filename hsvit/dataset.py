import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from glob import glob
from hsvit.preprocessor import skull_strip, apply_clahe, gaussian_smooth

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all .mat files.
            transform (callable, optional): Optional transforms to apply to tensors.
        """
        self.files = sorted(glob(os.path.join(root_dir, "*.mat")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = self.files[idx]
        with h5py.File(mat_path, 'r') as f:
            group = f['cjdata']
            image = np.array(group['image']).T
            label = int(np.array(group['label'])[0][0])

            # Preprocessing pipeline
            stripped = skull_strip(image)
            enhanced = apply_clahe(stripped)
            smoothed = gaussian_smooth(enhanced)

            # Normalize to [0, 1] and convert to tensor
            img_tensor = torch.tensor(smoothed / 255.0, dtype=torch.float32).unsqueeze(0)  # shape: (1, H, W)
            label_tensor = torch.tensor(label, dtype=torch.long)

            if self.transform:
                img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor