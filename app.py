from typing import List

import os
import gradio as gr
import numpy as np
from PIL import Image
import supervision as sv
import torch
from tqdm import tqdm
from utils.yolo_world import YOLOWorld
from utils import stastics
import timeit

from utils.efficient_sam import load, inference_with_boxes
from utils.video import (
    generate_file_name,
    calculate_end_frame_index,
    create_directory,
    remove_files_older_than
)

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
    ['./examples/shedding_concrete_0014.jpg', 'shedding_concrete', 0.1, 0.1, True, True, False, False],
    ['./examples/rusting_0044.jpg', 'rusting', 0.1, 0.1, True, True, False, False],
    ['./examples/rusting_0027.jpg', 'rusting', 0.1, 0.1, True, True, False, False],
    ['./examples/rusting_0138.jpg', 'shedding_concrete,rusting', 0.01, 0.1, True, True, False, False],
]
VIDEO_EXAMPLES = [
    ['./examples/output.mp4', 'shedding_concrete', 0.01, 0.2, False, False, False, False],
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(model_type='ti',  # model_type = 'ti' or 's'
                           ckpt='./weights-EfficientSAM/efficient_sam_vitt.pt',
                           device=DEVICE)
# 单病害单模型
# YOLO_WORLD_MODEL = {
#     "shedding_concrete": YOLOWorld(ckpt="./weights-YOLOWorld/shedding_concrete.pt",
#                                    device='0' if torch.cuda.is_available() else 'cpu'),
#     "rusting":  YOLOWorld(ckpt="./weights-YOLOWorld/rusting.pt",
#                           device='0' if torch.cuda.is_available() else 'cpu')
# }

# 多病害模型
YOLO_WORLD_MODEL = YOLOWorld(ckpt="./weights-YOLOWorld/shedding_concrete+rusting.pt",
                             device='0' if torch.cuda.is_available() else 'cpu')


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_scale=0.7)

# creating video results directory
create_directory(directory_path=RESULTS)


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def annotate_image(
    input_image: Image.Image,
    detections: sv.Detections,
    categories: List[str],
    with_label: bool = True,
    with_confidence: bool = False,
) -> Image.Image:
    if with_label:
        labels = [
            (
                f"{index + 1}.{categories[class_id]}: {confidence:.3f}"
                if with_confidence
                else f"{index + 1}.{categories[class_id]}"
            )
            for index, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence), start=0)
        ]
    else:
        labels = [
            (
                f"{index + 1}: {confidence:.3f}"
                if with_confidence
                else f"{index + 1}"
            )
            for index, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence), start=0)
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
    with_label: bool = True,
    with_confidence: bool = False,
    with_class_agnostic_nms: bool = False,
):
    start = timeit.default_timer()
    categories = process_categories(categories)
    # 单病害单模型
    # 未完成，策略：循环单模型检测所有病害，结果通过from_ultralytics封装Detections，将多Detections通过merge方法合并
    # results = YOLO_WORLD_MODEL[categories[0]].infer(input_image, confidence=confidence_threshold, iou=iou_threshold,
    #                                                 text=categories)
    # 多病害模型
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
        with_label=with_label,
        with_confidence=with_confidence
    )
    end = timeit.default_timer()
    total_num = stastics.get_total_num(detections)
    class_count = stastics.get_class_count(detections)
    total_seg_proportion = stastics.get_total_seg_proportion(detections)
    average_confidence = stastics.get_average_confidence(detections)
    duration = stastics.format_time(end - start)
    det_frame = stastics.get_det_dataFrame(detections)
    return output_image, total_num, class_count, total_seg_proportion, average_confidence, duration, det_frame


def process_video(
    input_video: str,
    categories: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_segmentation: bool = True,
    with_label: bool = True,
    with_confidence: bool = False,
    with_class_agnostic_nms: bool = False,
    progress=gr.Progress(track_tqdm=True)
):
    # cleanup of old video files
    remove_files_older_than(RESULTS, 30)

    categories = process_categories(categories)
    video_info = sv.VideoInfo.from_video_path(input_video)
    total = calculate_end_frame_index(input_video)
    frame_generator = sv.get_video_frames_generator(
        source_path=input_video,
        end=total
    )
    result_file_name = generate_file_name(extension="mp4")
    result_file_path = os.path.join(RESULTS, result_file_name)
    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for _ in tqdm(range(total), desc="Processing video..."):
            frame = next(frame_generator)
            frame = Image.fromarray(frame)
            results = YOLO_WORLD_MODEL.infer(frame, confidence=confidence_threshold, iou=iou_threshold, text=categories)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )
            frame_np = np.array(frame)
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=frame_np,
                    xyxy=detections.xyxy,
                    model=EFFICIENT_SAM_MODEL,
                    device=DEVICE
                )
            frame = annotate_image(
                input_image=frame,
                detections=detections,
                categories=categories,
                with_label=with_label,
                with_confidence=with_confidence
            )
            frame = np.array(frame)
            sink.write_frame(frame)
    return result_file_path


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.1,
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
    value=0.1,
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

with_label_component = gr.Checkbox(
    value=True,
    label="Display Label",
    info=(
        "Whether to display the detected objects' labels."
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
            with_label_component.render()
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
                value='shedding_concrete,rusting'
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

        with gr.Tab(label='result'):
            with gr.Row():
                with gr.Column(variant='compact'):
                    with gr.Group():
                        det_num_component = gr.Textbox(
                            label='Total number of targets',
                            placeholder='',
                            interactive=False
                        )
                        class_count_component = gr.DataFrame(
                            type='pandas',
                            headers=[''],
                            interactive=False
                        )
                with gr.Column():
                    total_seg_proportion_component = gr.Label(label='total seg')
                    with gr.Row():
                        average_confidence_component = gr.Label(label='average confidence')
                        total_duration_component = gr.Label(label='duration')

            with gr.Row():
                det_sheet_component = gr.DataFrame(
                    label='detection sheet',
                    type='pandas',
                    headers=['id', 'class_name', 'confidence', 'upper_left', 'lower_right'],
                    interactive=False
                )
        gr.Examples(
            fn=process_image,
            examples=IMAGE_EXAMPLES,
            run_on_click=True,
            # cache_examples=True,
            inputs=[
                input_image_component,
                image_categories_text_component,
                confidence_threshold_component,
                iou_threshold_component,
                with_segmentation_component,
                with_label_component,
                with_confidence_component,
                with_class_agnostic_nms_component
            ],
            outputs=[
                output_image_component,
                det_num_component,
                class_count_component,
                total_seg_proportion_component,
                average_confidence_component,
                total_duration_component,
                det_sheet_component
            ]
        )
    with gr.Tab(label="Video"):
        with gr.Row():
            input_video_component = gr.Video(
                label='Input Video'
            )
            output_video_component = gr.Video(
                label='Output Video'
            )
        with gr.Row():
            video_categories_text_component = gr.Textbox(
                label='Categories',
                placeholder='comma separated list of categories',
                scale=7,
                value='shedding_concrete,rusting'
            )
            video_submit_button_component = gr.Button(
                value='Submit',
                scale=1,
                variant='primary'
            )
            clear_video_button_component = gr.ClearButton(
                value='Clear',
                scale=1,
                variant='secondary',
                components=[input_video_component, output_video_component]
            )
        gr.Examples(
            fn=process_video,
            examples=VIDEO_EXAMPLES,
            # run_on_click=True,
            # cache_examples=True,
            inputs=[
                input_video_component,
                video_categories_text_component,
                confidence_threshold_component,
                iou_threshold_component,
                with_segmentation_component,
                with_label_component,
                with_confidence_component,
                with_class_agnostic_nms_component
            ],
            outputs=output_image_component
        )

    image_submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_label_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=[
            output_image_component,
            det_num_component,
            class_count_component,
            total_seg_proportion_component,
            average_confidence_component,
            total_duration_component,
            det_sheet_component
        ]
    )
    video_submit_button_component.click(
        fn=process_video,
        inputs=[
            input_video_component,
            video_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_segmentation_component,
            with_label_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=output_video_component
    )


demo.launch(debug=False, show_error=True)
