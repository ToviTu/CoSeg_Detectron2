import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math

src_dir = "/home/research/jianhong.t/OpenVocab_Seg_with_AutoRegres/src/"

class COCOStuffDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, img_size=224):
        """
        Args:
            image_dir (string): Directory with all the images.
            annotation_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = os.listdir(image_dir)
        self.img_size = img_size

        # Load the label mapping
        self.digit_to_object_mapping = {}
        with open(f'{src_dir}labels.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                self.digit_to_object_mapping[int(key)] = value.strip()
        self.digit_to_object_mapping[255] = "unlabeled"

    def center_crop(self, image, mask):
        transform = transforms.CenterCrop(self.img_size)
        return transform(image), transform(mask)

    def resize(self, image, mask):
        transform = transforms.CenterCrop(min(image.size))

        cropped_image = transform(image)
        cropped_mask = transform(mask)

        resized_image = transforms.Resize((self.img_size, self.img_size),transforms.InterpolationMode.BILINEAR)(cropped_image)
        resized_mask= transforms.Resize((self.img_size, self.img_size), transforms.InterpolationMode.NEAREST)(cropped_mask)
        return resized_image, resized_mask

    def __len__(self):
        return len(self.images)

    def get(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)#.convert('RGB')

        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))
        annotation = Image.open(annotation_name)

        ids = np.unique(np.array(annotation))
        labels = [self.digit_to_object_mapping[id] for id in ids]

        return image, annotation, labels

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name)#.convert('RGB')

        # Load annotation
        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))
        annotation = Image.open(annotation_name)

        image, mask = self.resize(image, annotation)
        mask = np.array(mask)
        mask += 1
        mask[mask==256] = 0


        # Indexed masks
        ids = np.unique(mask)
        ids = [id for id in ids if id != 0]
        nonempty_masks = [np.full(mask.shape, id) * (mask==id) for id in ids]
        nonempty_masks = sorted(nonempty_masks, key=lambda x: np.sum(x!=0), reverse=True)

        # Get ids and labels
        ids = [np.unique(mask)[-1] for mask in nonempty_masks]
        labels = [self.digit_to_object_mapping[id] for id in ids]

        # Convert to binary masks
        nonempty_masks = [(mask != 0).astype(float) for mask in nonempty_masks]

        sample = {'image': image, 'annotation': nonempty_masks, 'labels': labels, 'ids': ids}

        return sample

def collate_fn_factory(processor, max_size=20):

    def collate_fn(batch):
        size = processor.image_processor.size['shortest_edge'] #224
        transform = transforms.ToTensor()

        # Preprocess pixel values
        images = [each['image'] for each in batch]
        batch_pixel_values = processor(None, images=images, return_tensors='pt')['pixel_values']

        # Preprocess labels
        ids = torch.full((len(batch), max_size), 0)
        ids[:, :max_size] = torch.tensor([each['ids'][:max_size] + [0] * (max_size - len(each['ids'])) for each in batch])

        # Preprocess masks
        batch_masks = np.stack([
            np.stack(each['annotation'][:max_size] + [np.zeros((size, size))] * (max_size - len(each['annotation'])) )
            for each in batch
        ])
        batch_masks = torch.tensor(batch_masks[:, :max_size])

        return {
            "pixel_values": batch_pixel_values,
            "masks": batch_masks.type(torch.float32),
            "ids": ids.type(torch.long)
        }

    return collate_fn

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]