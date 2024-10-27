import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2

import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import random
from torchvision import transforms, utils

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        # Load image
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)
        
        # Remove alpha channel if it exists
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Convert keypoints to a NumPy array and reshape
        key_pts = self.key_pts_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        
        # Create sample dictionary
        sample = {'image': image, 'keypoints': key_pts}

        # Apply transformations if available
        if self.transform:
            sample = self.transform(sample)

        return sample

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        new_h, new_w = self.output_size

        # Resize image directly to the target size
        img = cv2.resize(image, (new_w, new_h))

        # Scale keypoints accordingly
        h, w = image.shape[:2]
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class Resize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = cv2.resize(image, (self.output_size[1], self.output_size[0]))
        return {'image': image, 'keypoints': keypoints}


import random
import numpy as np
import cv2

class GrayscaleJitter:
    """Apply brightness and contrast jitter to grayscale images."""

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        # Adjust brightness
        if self.brightness > 0:
            brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * brightness_factor, 0, 1)

        # Adjust contrast
        if self.contrast > 0:
            contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

        return {'image': image, 'keypoints': keypoints}



class RandomRotate:
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        # Generate a random angle for rotation
        angle = random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Rotate keypoints
        rotated_keypoints = self.rotate_keypoints(keypoints, center, angle)
        
        return {'image': rotated_image, 'keypoints': rotated_keypoints}

    def rotate_keypoints(self, keypoints, center, angle):
        """Rotate keypoints around a center point by a given angle."""
        angle_rad = np.deg2rad(angle)
        cos, sin = np.cos(angle_rad), np.sin(angle_rad)

        # Translate points to origin
        translated_points = keypoints - center

        # Apply rotation matrix
        rotated_points = np.empty_like(translated_points)
        rotated_points[:, 0] = translated_points[:, 0] * cos - translated_points[:, 1] * sin
        rotated_points[:, 1] = translated_points[:, 0] * sin + translated_points[:, 1] * cos

        # Translate points back
        rotated_points += center
        return rotated_points



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}