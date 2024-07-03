# -*- coding: utf-8 -*-
"""CV_archi_0.3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gr39W8G0bbFAYmhD0iv6oi0xzx68xR48
"""

import torch
from torch import nn
from transformers import CLIPProcessor
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import tqdm

from coseg.model.model import CoSeg
from coseg.model.legacy_model import CoSeg_legacy
from coseg.model.lang_model import CLIPLang

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""## Dataset"""

src_dir = "/home/research/jianhong.t/OpenVocab_Seg_with_AutoRegres/src/"
dataset_dir = "/scratch/t.tovi/datasets/"
image_dir = "coco-stuff/COCO_stuff_images/train2017/"
annotation_dir = "COCO_stuff_annotations/train2017/"

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

"""## Define the collate function"""

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

"""## Training Pipeline"""

def dice_loss(y_true, y_pred):
    numerator = 2 * torch.sum(y_true * y_pred)
    denominator = torch.sum(y_true + y_pred)
    return 1 - numerator / denominator

device = 0

# Define dataset dir
dataset_dir = "/scratch/t.tovi/datasets/"

# Create dataset object
data = COCOStuffDataset(
    dataset_dir+image_dir,
    dataset_dir+annotation_dir,
    img_size=224
)

lang_model = CLIPLang()
lang_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Get loss query table

labels = data.digit_to_object_mapping

label_indices = list(data.digit_to_object_mapping.keys())
label_text = ["a photo of " + data.digit_to_object_mapping[each] for each in label_indices]
inputs = processor(label_text, padding=True, return_tensors='pt')

with torch.no_grad():
    label_embeddings = lang_model(**inputs)["text_embeds"]
label_embeddings.requires_grad_(False)

reverse_mapping = {v: k for k, v in data.digit_to_object_mapping.items()}

# Get the collate function

collate_fn = collate_fn_factory(processor)

# Create batch data loader
data_loader = DataLoader(data, batch_size=32, collate_fn=collate_fn, num_workers=4, shuffle=True)

# Initialize the model
model = CoSeg_legacy(d_reduce=128, nencoder=4, ndecoder=4) #CoSeg(d_reduce=128, nencoder=4, ndecoder=4, lang_model='xatten')
m = nn.Sigmoid()
model.to(device)

# Freeze all
# model.encoders.vision_encoder.requires_grad_(False)
# model.encoders.vision_projector.requires_grad_(False)

encoder_params = [
    model.encoders.query_embeddings,
    model.encoders.decoder,
]

decoder_params = [
    model.reduces,
    model.film_mul,
    model.film_add,
    model.decoder,
    model.mask_head
]

for param in encoder_params + decoder_params:
    param.requires_grad_(True)

# Define training parameters
lr_encoder = 1e-4
lr_decoer = 1e-4
alpha = 0.08
beta = 0.18
temperature = 0.08
num_epochs = 100

# Optimizer
# optim = AdamW(
#     [param for param in model.parameters() if param.requires_grad],
#     weight_decay = 1e-4
# )
optim = AdamW(
    [
        {'params': param.parameters(), "lr" : lr_encoder}
        for param in encoder_params
    ] +\
    [
        {'params': param.parameters(), "lr" : lr_decoer}
        for param in decoder_params
    ],
    weight_decay = 1e-4
)

scheduler = CosineAnnealingLR(optim, T_max=len(data_loader), eta_min=1e-6)
#scheduler = StepLR(optim, step_size=10, gamma=0.7)

# Loss
mask_objective = nn.BCELoss()
mask_objective2 = dice_loss
lang_objective = nn.CrossEntropyLoss()

"""## Train"""
label_embeddings = label_embeddings.to(0)
label_embeddings = F.normalize(label_embeddings, dim=-1)

count = 0
for _ in range(num_epochs):
    batch_loss = 0
    batch_l1 = 0
    batch_l2 = 0
    batch_l3 = 0
    for batch in data_loader:
        # Prepare data
        pixel_values = batch['pixel_values'].to(device)
        masks = batch['masks'].to(device)
        ids = batch['ids'].to(device)

        mask_logits, pred_embeddings = model(pixel_values)
        pred_embeddings = F.normalize(pred_embeddings, dim=-1)
        label_logits = pred_embeddings @ label_embeddings.T / temperature

        # Compute loss
        mask_prob = m(mask_logits)
        l1 = mask_objective(mask_prob, masks)
        l2 = alpha * lang_objective(label_logits.permute(0, 2, 1), ids)
        l3 = beta * mask_objective2(masks, mask_prob)

        # Total loss
        loss = l1 + l2 + l3

        loss.backward()
        optim.step()
        optim.zero_grad()

        batch_loss += loss.detach().cpu().item()
        batch_l1 += l1.detach().cpu().item()
        batch_l2 += l2.detach().cpu().item()
        batch_l3 += l3.detach().cpu().item()

        if (count+1) % 64 == 0:
            print(f"Iter: {count} Avrage batch loss: {batch_loss / 64}, {batch_l1 / 64}, {batch_l2 / 64}, {batch_l3 / 64}")
            batch_loss = 0
            batch_l1 = 0
            batch_l2 = 0
            batch_l3 = 0

        count += 1

    scheduler.step()
    print("One training epoch done")
    torch.save(model.state_dict(), "/scratch/t.tovi/autoseg_v0.3")