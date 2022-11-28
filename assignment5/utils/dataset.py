import json
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .helper import _download, _tar_extract


class VOC2007DetectionTiny(Dataset):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """
    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"

    def __init__(
            self,
            dataset_dir: str,
            split: str = "train",
            image_size: int = 224,
    ):
        """
        Args:
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. For
                val, center crop will not be taken to capture all detections.
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "VOCdevkit/VOC2007/JPEGImages"
        self.image_size = image_size
        self.split = split

        # Download the dataset if it doesn't exist.
        self.download()

        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        # Make a (class to ID) and inverse (ID to class) mapping.
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        # Load instances from JSON file:
        with open(self.dataset_dir / f'voc07_{split}.json') as f:
            self.instances = json.load(f)

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        self.image_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def __str__(self):
        return f"VOC2007DetectionTiny({self.split}): size={len(self)}"

    def __len__(self):
        """ Return the number of images in the dataset. """
        return len(self.instances)

    def download(self):
        """Download the dataset."""
        if not self.image_dir.exists():
            print("Downloading dataset...")
            _tar_extract(_download(self.URL, self.dataset_dir), remove=True)
            print("Done!")

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Return a single data point. It is composed of an image with shape (C, H, W) and its bounding boxes with shape
        (N, 5). The 5 elements of each bounding box are (x1, y1, x2, y2, class_id).
        """

        # PIL image and dictionary of annotations.
        image_name, ann = self.instances[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")

        # Collect a list of GT boxes: (N, 4), and GT classes: (N, )
        gt_boxes = torch.tensor([inst["xyxy"] for inst in ann])
        gt_classes = torch.tensor([self.class_to_idx[inst["name"]] for inst in ann])
        gt_classes = gt_classes.unsqueeze(1)  # (N, 1)

        # Record original image size before transforming.
        original_width, original_height = image.size

        # Normalize bounding box co-ordinates to bring them in [0, 1]. This is
        # temporary, simply to ease the transformation logic.
        normalize_tens = torch.tensor(
            [original_width, original_height, original_width, original_height]
        )
        gt_boxes /= normalize_tens[None, :]

        # Transform input image to CHW tensor.
        image = self.image_transform(image)

        # WARN: Even dimensions should be even numbers else it messes up
        # upsampling in FPN.

        # Apply image resizing transformation to bounding boxes.
        if self.image_size is not None:
            if original_height >= original_width:
                new_width = self.image_size
                new_height = original_height * self.image_size / original_width
            else:
                new_height = self.image_size
                new_width = original_width * self.image_size / original_height

            _x1 = (new_width - self.image_size) // 2
            _y1 = (new_height - self.image_size) // 2

            # Un-normalize bounding box co-ordinates and shift due to center crop.
            # Clamp to (0, image size).
            gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
            gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
            gt_boxes[:, 2] = torch.clamp(
                gt_boxes[:, 2] * new_width - _x1, max=self.image_size
            )
            gt_boxes[:, 3] = torch.clamp(
                gt_boxes[:, 3] * new_height - _y1, max=self.image_size
            )

        # Concatenate GT classes with GT boxes; shape: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)

        # Center cropping may completely exclude certain boxes that were close
        # to image boundaries. Set them to -1
        invalid = (gt_boxes[:, 0] > gt_boxes[:, 2]) | (
                gt_boxes[:, 1] > gt_boxes[:, 3]
        )
        gt_boxes[invalid] = -1

        # Pad to max 40 boxes, that's enough for VOC.
        gt_boxes = torch.cat(
            [gt_boxes, torch.zeros(40 - len(gt_boxes), 5).fill_(-1.0)]
        )
        # Return image path because it is needed for evaluation.
        return str(image_path), image, gt_boxes
