from torch.utils.data import Dataset
import pydicom
import os
import glob
import numpy as np
import torch

class ProstateXDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_dirs = sorted([
            os.path.join(root_dir, 'ProstateX_orig', 'manifest-1605042674814', 'PROSTATEx', d)
            for d in os.listdir(os.path.join(root_dir, 'ProstateX_orig', 'manifest-1605042674814', 'PROSTATEx'))
            if os.path.isdir(os.path.join(root_dir, 'ProstateX_orig', 'manifest-1605042674814', 'PROSTATEx', d))
        ])

        self.idx_to_class = {
            0: 'peripheral zone',
            1: 'transition zone'
        }

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # MRI
        dir1_mri = next(d for d in os.listdir(patient_dir) if not d.startswith('.'))
        dir2_mri = next(d for d in os.listdir(os.path.join(patient_dir, dir1_mri)) if not d.startswith('.'))
        mri_path = os.path.join(patient_dir, dir1_mri, dir2_mri)
        mri_images = sorted(glob.glob(os.path.join(mri_path, '*.dcm')), key=lambda x: int(x.split('/')[-1].split('-')[1].split('.')[0]))
        mri_images = np.stack([pydicom.dcmread(img).pixel_array.astype(np.float32) for img in mri_images])

        # Segmentation
        seg_dir = patient_dir.replace('ProstateX_orig', 'ProstateX_seg')
        dir1_seg = next(d for d in os.listdir(seg_dir) if not d.startswith('.'))
        dir2_seg = next(d for d in os.listdir(os.path.join(seg_dir, dir1_seg)) if not d.startswith('.'))
        seg_path = os.path.join(seg_dir, dir1_seg, dir2_seg, '1-1.dcm')
        seg_image = pydicom.dcmread(seg_path).pixel_array.astype(np.float32)

        # Reshape the segmentation to [4, num_slices, H, W] (Assume seg_image is (4*num_slices, H, W)
        num_slices = mri_images.shape[0]
        seg_image = seg_image.reshape(4, num_slices, *seg_image.shape[1:])
        # seg_image = seg_image[:len(self.idx_to_class)] # Slice out the first two classes (prostate & transition zones)
        seg_image[2] = 1 - seg_image[0] - seg_image[1]
        seg_image = seg_image[:3] # Slice out the first two classes (prostate & transition zones)


        mri_tensor = torch.tensor(mri_images, dtype=torch.float32)
        seg_tensor = torch.tensor(seg_image, dtype=torch.long)
        if self.transform:
            mri_tensor = self.transform(mri_tensor)
            seg_tensor = self.transform(seg_tensor)
        return mri_tensor, seg_tensor