import os
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import Union, Dict, Optional
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import Image
from matplotlib.patches import Rectangle
from torch import optim
from tqdm import tqdm


def hello_helper():
    print("Hello from helper.py!")


def _download(url: str, root: Union[str, Path]) -> Path:
    """Download a file from a URL to a given root directory."""
    target = Path(root) / url.split("/")[-1]

    with urlopen(url) as source, open(target, "wb") as output:
        size = int(source.info().get("Content-Length"))
        with tqdm(total=size, ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))
    return target


def _tar_extract(file: Union[str, Path], remove: bool = True):
    """Extract a tar file."""
    file = Path(file)
    with tarfile.open(file) as tar:
        tar.extractall(file.parent)
    if remove:
        file.unlink()


def detection_visualizer(img: Union[torch.Tensor, np.array, Image],
                         idx_to_class: Dict,
                         bbox: Optional[torch.Tensor] = None,
                         pred: Optional[torch.Tensor] = None,
                         points: Optional[torch.Tensor] = None,
                         figsize: Optional[tuple] = None):
    """
    Data visualizer on the original image. Support both GT
    box input and proposal input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates
            (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional),
            a tensor of shape N'x6, where N' is the number
            of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    # Convert image to HWC if it is passed as a Tensor (0-1, CHW).
    if isinstance(img, torch.Tensor):
        img = (img * 255).permute(1, 2, 0)

    img_copy = np.array(img).astype("uint8")
    _, ax = plt.subplots(frameon=False, figsize=figsize)

    ax.axis("off")
    ax.imshow(img_copy)

    if points is not None:
        points_x = [t[0] for t in points]
        points_y = [t[1] for t in points]
        ax.scatter(points_x, points_y, color="yellow", s=24)

    if bbox is not None:
        for single_bbox in bbox:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(1.0, 0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                ax.text(
                    x0, y0, obj_cls, size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    if pred is not None:
        for single_bbox in pred:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(0, 1.0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                conf_score = single_bbox[5].item()
                ax.text(
                    x0, y0 + 15, f"{obj_cls}, {conf_score:.2f}",
                    size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    plt.tight_layout()
    plt.show()


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def rel_error(x, y, eps=1e-10):
    """
    Compute the relative error between a pair of tensors x and y,
    which is defined as:

                            max_i |x_i - y_i]|
    rel_error(x, y) = -------------------------------
                      max_i |x_i| + max_i |y_i| + eps

    Inputs:
    - x, y: Tensors of the same shape
    - eps: Small positive constant for numeric stability

    Returns:
    - rel_error: Scalar giving the relative error between x and y
    """
    """ returns relative error between x and y """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


def train_detector(
        detector,
        train_loader,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-4,
        max_iters: int = 5000,
        log_period: int = 20,
        device: str = "cpu",
):
    """
    Train the detector. We use SGD with momentum and step decay.
    """

    detector.to(device=device)

    # Optimizer: use SGD with momentum.
    # Use SGD with momentum:
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        momentum=0.9,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # LR scheduler: use step decay at 70% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Keep track of training loss for plotting.
    loss_history = []

    train_loader = infinite_loader(train_loader)
    detector.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        _, images, gt_boxes = next(train_loader)

        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # Dictionary of loss scalars.
        losses = detector(images, gt_boxes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()


def inference_with_detector(
        detector,
        test_loader,
        idx_to_class,
        score_thresh: float,
        nms_thresh: float,
        output_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
):
    # ship model to GPU
    detector.to(dtype=dtype, device=device)

    detector.eval()
    start_t = time.time()

    # Define an "inverse" transform for the image that un-normalizes by ImageNet
    # color. Without this, the images will NOT be visually understandable.
    inverse_norm = T.Compose(
        [
            T.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            T.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    if output_dir is not None:
        det_dir = "mAP/input/detection-results"
        gt_dir = "mAP/input/ground-truth"
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        os.mkdir(det_dir)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.mkdir(gt_dir)

    for iter_num, test_batch in enumerate(test_loader):
        image_paths, images, gt_boxes = test_batch
        images = images.to(dtype=dtype, device=device)

        with torch.no_grad():
            if score_thresh is not None and nms_thresh is not None:
                # shapes: (num_preds, 4) (num_preds, ) (num_preds, )
                pred_boxes, pred_classes, pred_scores = detector(
                    images,
                    test_score_thresh=score_thresh,
                    test_nms_thresh=nms_thresh,
                )

        # Skip current iteration if no predictions were found.
        if pred_boxes.shape[0] == 0:
            continue

        # Remove padding (-1) and batch dimension from predicted / GT boxes
        # and transfer to CPU. Indexing `[0]` here removes batch dimension:
        gt_boxes = gt_boxes[0]
        valid_gt = gt_boxes[:, 4] != -1
        gt_boxes = gt_boxes[valid_gt].cpu()

        valid_pred = pred_classes != -1
        pred_boxes = pred_boxes[valid_pred].cpu()
        pred_classes = pred_classes[valid_pred].cpu()
        pred_scores = pred_scores[valid_pred].cpu()

        image_path = image_paths[0]
        # Un-normalize image tensor for visualization.
        image = inverse_norm(images[0]).cpu()

        # Combine predicted classes and scores into boxes for evaluation
        # and visualization.
        pred_boxes = torch.cat(
            [pred_boxes, pred_classes.unsqueeze(1), pred_scores.unsqueeze(1)], dim=1
        )

        # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
        if output_dir is not None:
            file_name = os.path.basename(image_path).replace(".jpg", ".txt")
            with open(os.path.join(det_dir, file_name), "w") as f_det, open(
                    os.path.join(gt_dir, file_name), "w"
            ) as f_gt:
                for b in gt_boxes:
                    f_gt.write(
                        f"{idx_to_class[b[4].item()]} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
                for b in pred_boxes:
                    f_det.write(
                        f"{idx_to_class[b[4].item()]} {b[5]:.6f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
        else:
            detection_visualizer(
                image, idx_to_class, gt_boxes, pred_boxes
            )

    end_t = time.time()
    print(f"Total inference time: {end_t - start_t:.1f}s")
