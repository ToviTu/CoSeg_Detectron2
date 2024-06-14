import detectron2.data.transforms as T
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from transformers import CLIPProcessor
import copy
import torch
import json
import numpy as np

class TrainMapper:

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_value,
        max_seq_len
    ):
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.tf_gens = augmentations
        self.is_train = is_train
        self.image_format = image_format
        self.ignore_value = ignore_value
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(self, cfg, is_train=True):

        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.INPUT.IGNORE_VALUE,
            )
        ]

        ret = {
            'is_train': is_train,
            'augmentations': augs,
            'image_format': cfg.INPUT.FORMAT,
            'ignore_value': cfg.INPUT.IGNORE_VALUE,
            'max_seq_len': cfg.MODEL.MAX_SEQ_LEN
        }

        return ret

    def __call__(self, dataset_dict):
        # Load Image & Annotation
        # Crop Image & Resize
        # Process with CLIP processor

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        # For debugging
        #dataset_dict['raw_image'] = image
        
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")

        original_size = image.shape[:2]
        shorter_size = min(original_size)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tf_gens, aug_input)

        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        image = self.processor(None, images=image, return_tensors='pt', num_workers=1)['pixel_values'].squeeze(0)

        # Split gt into an array of masks
        class_ids = torch.unique(sem_seg_gt)
        masks = []
        for id in class_ids:
            if id == 255: continue # Skip empty mask
            mask = (torch.full(sem_seg_gt.shape, id) * (sem_seg_gt == id))
            masks.append(mask)
        masks = sorted(masks, key=lambda x: torch.sum(x != 0), reverse=True)

        # get indices
        ids = [torch.unique(mask)[-1].item() for mask in masks]
        masks = [(mask != 0).float() for mask in masks]

        # pad / truncate the sequences
        assert len(ids) == len(masks), "Inconsistent sequence length!"
        ids = ids[:self.max_seq_len]
        masks = masks[:self.max_seq_len]

        ids += [self.ignore_value] * (self.max_seq_len - len(ids))
        masks += [torch.full(sem_seg_gt.shape, 0.)] * (self.max_seq_len - len(masks))

        # Form the final mask sequence
        masks = torch.stack(masks, dim=0)

        dataset_dict['ids'] = ids
        dataset_dict['image'] = image
        dataset_dict['masks'] = masks
        dataset_dict['width'], dataset_dict['height'] = image.shape[-2:]

        return dataset_dict