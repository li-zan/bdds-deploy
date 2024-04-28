from typing import List

import os
import gradio as gr
import numpy as np
from PIL import Image
import supervision as sv
import torch
from tqdm import tqdm
from utils.yolo_world import YOLOWorld

from utils.efficient_sam import load, inference_with_boxes

MARKDOWN = """
# YOLO-World + EfficientSAM 🔥



This is a demo of zero-shot object detection and instance segmentation using 
[YOLO-World](https://github.com/AILab-CVC/YOLO-World) and 
[EfficientSAM](https://github.com/yformer/EfficientSAM).

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision).
"""

RESULTS = "results"

IMAGE_EXAMPLES = [

]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(model_type='ti',  # model_type = 'ti' or 's'
                           ckpt='./weights-EfficientSAM/efficient_sam_vitt.pt',
                           device=DEVICE)
YOLO_WORLD_MODEL = YOLOWorld(ckpt="./weights-YOLOWorld/200_best_01.pt",
                             device='0' if torch.cuda.is_available() else 'cpu')


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def annotate_image(
    input_image: Image.Image,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
) -> Image.Image:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_image(
    input_image: Image.Image,
    categories: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_segmentation: bool = True,
    with_confidence: bool = False,
    with_class_agnostic_nms: bool = False,
) -> Image.Image:

    categories = process_categories(categories)
    results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold, iou=iou_threshold, text=categories)
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(
        class_agnostic=with_class_agnostic_nms,
        threshold=iou_threshold
    )
    input_image_np = np.array(input_image)
    if with_segmentation:
        detections.mask = inference_with_boxes(
            image=input_image_np,
            xyxy=detections.xyxy,
            model=EFFICIENT_SAM_MODEL,
            device=DEVICE
        )
    output_image = annotate_image(
        input_image=input_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence
    )
    return output_image


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.01,
    step=0.01,
    label="Confidence Threshold",
    info=(
        "The confidence threshold for the YOLO-World model. Lower the threshold to "
        "reduce false negatives, enhancing the model's sensitivity to detect "
        "sought-after objects. Conversely, increase the threshold to minimize false "
        "positives, preventing the model from identifying objects it shouldn't."
    ))

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.005,
    step=0.01,
    label="IoU Threshold",
    info=(
        "The Intersection over Union (IoU) threshold for non-maximum suppression. "
        "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
        "making the detection process stricter. On the other hand, increase the value "
        "to allow more overlapping bounding boxes, accommodating a broader range of "
        "detections."
    ))

with_segmentation_component = gr.Checkbox(
    value=True,
    label="With Segmentation",
    info=(
        "Whether to run EfficientSAM for instance segmentation."
    )
)

with_confidence_component = gr.Checkbox(
    value=False,
    label="Display Confidence",
    info=(
        "Whether to display the confidence of the detected objects."
    )
)

with_class_agnostic_nms_component = gr.Checkbox(
    value=False,
    label="Use Class-Agnostic NMS",
    info=(
        "Suppress overlapping bounding boxes across all classes."
    )
)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Accordion("Configuration", open=False):
        confidence_threshold_component.render()
        iou_threshold_component.render()
        with gr.Row():
            with_segmentation_component.render()
            with_confidence_component.render()
            with_class_agnostic_nms_component.render()
    with gr.Tab(label="Image"):
        with gr.Row():
            input_image_component = gr.Image(
                type='pil',
                label='Input Image'
            )
            output_image_component = gr.Image(
                type='pil',
                label='Output Image'
            )
        with gr.Row():
            image_categories_text_component = gr.Textbox(
                label='Categories',
                placeholder='comma separated list of categories',
                scale=7,
                value='shedding_concrete'
            )
            image_submit_button_component = gr.Button(
                value='Submit',
                scale=1,
                variant='primary'
            )
            clear_image_button_component = gr.ClearButton(
                value='Clear',
                scale=1,
                variant='secondary',
                components=[input_image_component, output_image_component]
            )

    image_submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_image_component
    )


demo.launch(debug=False, show_error=True)
