import torch
import numpy as np
from torchvision.transforms import ToTensor
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from efficient_sam.build_efficient_sam import build_efficient_sam_vits


def load(model_type, ckpt, device: torch.device):
    if model_type == "ti":
        model = build_efficient_sam_vitt(ckpt)
    elif model_type == "s":
        model = build_efficient_sam_vits(ckpt)
    return model.to(device)


def inference_with_point(
    image: np.ndarray,
    point: np.ndarray,
    model,
    device: torch.device
) -> np.ndarray:
    t_point = torch.reshape(torch.tensor(point), [1, 1, 1, 2])
    t_label = torch.reshape(torch.tensor([1]), [1, 1, 1])
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        t_point.to(device),
        t_label.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


def inference_with_box(
    image: np.ndarray,
    box: np.ndarray,
    model,
    device: torch.device
) -> np.ndarray:
    bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        bbox.to(device),
        bbox_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


def inference_with_boxes(
    image: np.ndarray,
    xyxy: np.ndarray,
    model,
    device: torch.device
) -> np.ndarray:
    masks = []
    for [x_min, y_min, x_max, y_max] in xyxy:
        box = np.array([[x_min, y_min], [x_max, y_max]])
        mask = inference_with_box(image, box, model, device)
        masks.append(mask)
    return np.array(masks)


def inference_with_points(
    image: np.ndarray,
    points: [],
    model,
    device: torch.device
) -> np.ndarray:
    masks = []
    for [x, y] in points:
        point = np.array([x, y])
        mask = inference_with_point(image, point, model, device)
        masks.append(mask)
    return np.array(masks)
