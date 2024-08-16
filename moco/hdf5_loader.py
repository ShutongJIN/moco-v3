import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image 

class HDF5Dataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Path to the folder containing hdf5 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform  # This will be TwoCropsTransform for MoCo
        self.image_offsets = []
        self.total_images = 0
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from HDF5 files to build index of images."""
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith('.hdf5'):
                file_path = os.path.join(self.folder_path, file)
                with h5py.File(file_path, 'r') as hdf:
                    for video_id in hdf.keys():
                        video_group = hdf[video_id]
                        for suffix in video_group.keys():
                            suffix_group = video_group[suffix]
                            for dataset_name in suffix_group.keys():
                                dataset = suffix_group[dataset_name]
                                num_images = len(dataset)
                                self.image_offsets.append((file_path, video_id, suffix, dataset_name, self.total_images, self.total_images + num_images))
                                self.total_images += num_images
        print(f"Total images: {self.total_images}")

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        for file, video_id, suffix, dataset_name, start_idx, end_idx in self.image_offsets:
            if start_idx <= idx < end_idx:
                with h5py.File(file, 'r') as hdf:
                    image_idx = idx - start_idx
                    dataset = hdf[video_id][suffix][dataset_name]
                    image = dataset[image_idx]
                    break

        image = torch.from_numpy(image).permute(2, 0, 1).float() 

        if self.transform:
            image = transforms.ToPILImage()(image) 
            image1, image2 = self.transform(image) 
            images = self.transform(image) 
            stacked_images = torch.stack([image1, image2], dim=0)
            return stacked_images  # Return the stacked tensor with shape [2, C, H, W]

        return image

    def close(self):
        pass
