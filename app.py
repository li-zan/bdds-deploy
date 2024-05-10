from typing import List

import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
import torch
from tqdm import tqdm
from utils.yolo_world import YOLOWorld
from utils import stastics
import timeit

from utils.efficient_sam import load, inference_with_boxes, inference_with_points
from utils.video import (
    generate_file_name,
    calculate_end_frame_index,
    create_directory,
    remove_files_older_than
)
TITLE = "Bridge Defect Detection"
MARKDOWN = """
<h1 style="text-align:center;display:block">YOLO-World + EfficientSAM 🔥</h1>



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
    ['./examples/output.mp4', 'shedding_concrete,rusting', 0.1, 0.1, False, True, False, False],
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


def get_points_with_draw(image: Image.Image, points: [], seg_refine_mode, with_segmentation: bool, evt: gr.SelectData):
    if points is None:
        points = []
    if with_segmentation is False:
        info("segmentation is not enable in configuration")
        return image, points
    print("Starting draw point")
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0)
    points.append([x, y])

    if image is not None:
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
        )
    print("1-points=", points)
    if seg_refine_mode == 'Box':
        for i in range(0, len(points), 2):
            if len(points) >= i + 2:
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x1 < x2 and y1 < y2:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                elif x1 < x2 and y1 >= y2:
                    draw.rectangle([x1, y2, x2, y1], outline="red", width=5)
                    points[i][1] = y2
                    points[i + 1][1] = y1
                elif x1 >= x2 and y1 < y2:
                    draw.rectangle([x2, y1, x1, y2], outline="red", width=5)
                    points[i][0] = x2
                    points[i + 1][0] = x1
                elif x1 >= x2 and y1 >= y2:
                    draw.rectangle([x2, y2, x1, y1], outline="red", width=5)
                    points[i][0] = x2
                    points[i][1] = y2
                    points[i + 1][0] = x1
                    points[i + 1][1] = y1
    print("2-points=", points)
    return image, points


def refine_segmentation(
    input_image: Image.Image,
    output_image: Image.Image,
    points: [],
    with_segmentation: bool = True,
    with_label: bool = True,
    with_confidence: bool = True
):
    if points is None or len(points) == 0:
        return output_image, points
    if with_segmentation is False:
        info("segmentation is not enable in configuration")
        return output_image, points
    print("Starting refine segmentation")
    # 剔除检测框外的点
    global Detections, Categories
    detections = Detections
    categories = Categories

    filtered_points = []
    filtered_indices = []
    for point in points:
        x, y = point
        max_area = 0
        max_area_index = -1
        for i, detection in enumerate(detections.xyxy):
            x1, y1, x2, y2 = detection[:4]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_area_index = i
        if max_area_index != -1:
            filtered_points.append(point)
            filtered_indices.append(max_area_index)
    if len(filtered_points) == 0:
        info("All points are invalid")
        return output_image, points
    # 根据point做seg
    input_image_np = np.array(input_image)
    point_mask = inference_with_points(
        image=input_image_np,
        points=filtered_points,
        model=EFFICIENT_SAM_MODEL,
        device=DEVICE
    )
    # 更新mask
    updated_mask = detections.mask.copy()
    for i, index in enumerate(filtered_indices):
        updated_mask[index] = np.logical_or(updated_mask[index], point_mask[i])
    detections.mask = updated_mask

    output_image = annotate_image(
        input_image=input_image,
        detections=detections,
        categories=categories,
        with_label=with_label,
        with_confidence=with_confidence
    )
    points = []
    return output_image, points


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
    total_seg_proportion = stastics.get_total_seg_proportion(detections, input_image)
    average_confidence = stastics.get_average_confidence(detections)
    duration = stastics.format_time(end - start)
    det_frame = stastics.get_det_dataFrame(detections, input_image)
    global Detections, Categories
    Detections = detections
    Categories = categories
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
            frame = frame[:, :, ::-1]
            frame = Image.fromarray(frame, mode="RGB")
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
            frame = frame[:, :, ::-1]
            cv2.namedWindow("Real-Time Inference", 0)
            cv2.imshow("Real-Time Inference", frame)
            cv2.waitKey(10)
            sink.write_frame(frame)
        cv2.destroyWindow("Real-Time Inference")
    return result_file_path


def info(msg):
    gr.Info(msg)


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.002,
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


with gr.Blocks(title=TITLE) as demo:
    global_points = gr.State([])
    Detections = None
    Categories = None
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
            with gr.Column():
                with gr.Group():
                    seg_refine_mode_component = gr.Radio(
                        choices=['Point', 'Box'],
                        label='Refine Mode',
                        value='Point',
                        scale=1,
                        interactive=True
                    )
                    seg_refine_button_component = gr.Button(
                        value='Refine',
                        scale=1,
                        variant='secondary'
                    )
            clear_image_button_component = gr.ClearButton(
                value='Clear',
                scale=1,
                variant='secondary',
                components=[input_image_component, output_image_component, global_points]
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

    output_image_component.select(
        fn=get_points_with_draw,
        inputs=[output_image_component, global_points, seg_refine_mode_component, with_segmentation_component],
        outputs=[output_image_component, global_points]
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
    seg_refine_button_component.click(
        fn=refine_segmentation,
        inputs=[
            input_image_component,
            output_image_component,
            global_points,
            with_segmentation_component,
            with_label_component,
            with_confidence_component,
        ],
        outputs=[
            output_image_component,
            global_points,
            # det_num_component,
            # class_count_component,
            # total_seg_proportion_component,
            # average_confidence_component,
            # total_duration_component,
            # det_sheet_component
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
