from typing import List

import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
import torch
from tqdm import tqdm
from model.yolo_world import YOLOWorld
from utils import statistics
import timeit

from model.efficient_sam import load, inference_with_boxes, inference_with_points
from utils.history import History
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


def get_points_with_draw(image: Image.Image, points: [], refine_mode, with_segmentation: bool, evt: gr.SelectData):
    if points is None:
        points = []
    if with_segmentation is False and refine_mode == 'Point':
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
    if refine_mode == 'Box':
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
    return image, points


def refine(
    input_image: Image.Image,
    output_image: Image.Image,
    confidence_threshold: float,
    iou_threshold: float,
    points: [],
    refine_mode: str,
    total_seg_proportion,
    average_confidence,
    det_num,
    class_count,
    det_frame,
    with_segmentation: bool = True,
    with_label: bool = True,
    with_confidence: bool = True,
    with_class_agnostic_nms: bool = False
):
    if points is None or len(points) == 0:
        return output_image, points, total_seg_proportion, average_confidence, det_num, class_count, det_frame
    if refine_mode == 'Point' and with_segmentation is False:
        info("segmentation is not enable in configuration")
        return output_image, points, total_seg_proportion, average_confidence, det_num, class_count, det_frame
    if refine_mode == 'Box':
        if len(points) % 2 != 0:
            info("The coordinates of box are incomplete")
            return output_image, points, total_seg_proportion, average_confidence, det_num, class_count, det_frame
    valid = True
    if refine_mode == 'Point':
        output_image, total_seg_proportion, det_frame, valid = \
            refine_segmentation(input_image, output_image, points, total_seg_proportion,
                                det_frame, with_label, with_confidence)
    if refine_mode == 'Box':
        output_image, total_seg_proportion, average_confidence, det_num, class_count, det_frame = \
            refine_detection(input_image, confidence_threshold, iou_threshold,
                             points, with_segmentation, with_label, with_confidence, with_class_agnostic_nms)
    if valid:
        global Detections
        history.add_entry(output_image, total_seg_proportion, average_confidence, det_num, class_count, det_frame,
                          Detections)
    points = []
    return output_image, points, total_seg_proportion, average_confidence, det_num, class_count, det_frame


def refine_detection(
    input_image: Image.Image,
    confidence_threshold: float,
    iou_threshold: float,
    points: [],
    with_segmentation: bool = True,
    with_label: bool = True,
    with_confidence: bool = True,
    with_class_agnostic_nms: bool = False
):
    print("Starting refine detection")

    global Detections, Categories
    detections = Detections
    categories = Categories

    for i in range(0, len(points), 2):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        region = input_image.crop((x1, y1, x2, y2))
        region_detections = YOLO_WORLD_MODEL.infer(
            region,
            confidence=confidence_threshold,
            iou=iou_threshold,
            text=categories,
            class_agnostic=with_class_agnostic_nms
        )
        # 将检测结果调整回全图坐标系
        for detection in region_detections.xyxy:
            detection[0] += x1
            detection[1] += y1
            detection[2] += x1
            detection[3] += y1
        # 移除与现有检测结果重复的目标
        keep_indices = []
        for j, region_detection in enumerate(region_detections.xyxy):
            is_duplicate = False
            for detection in detections.xyxy:
                if statistics.calculate_iou(region_detection, detection) > 0.4:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(j)

        # 根据keep_indices更新region_detections的所有属性
        if len(keep_indices) > 0:
            region_detections.xyxy = region_detections.xyxy[keep_indices]
            if region_detections.mask is not None:
                region_detections.mask = region_detections.mask[keep_indices]
            if region_detections.confidence is not None:
                region_detections.confidence = region_detections.confidence[keep_indices]
            if region_detections.class_id is not None:
                region_detections.class_id = region_detections.class_id[keep_indices]
            if region_detections.tracker_id is not None:
                region_detections.tracker_id = region_detections.tracker_id[keep_indices]
            for key in region_detections.data:
                if isinstance(region_detections.data[key], np.ndarray):
                    region_detections.data[key] = region_detections.data[key][keep_indices]
                else:
                    region_detections.data[key] = [region_detections.data[key][k] for k in keep_indices]
        else:
            region_detections = sv.Detections(xyxy=np.array([[0, 0, 0, 0]]))
            region_detections.xyxy = []

        input_image_np = np.array(input_image)
        if with_segmentation:
            region_detections.mask = inference_with_boxes(
                image=input_image_np,
                xyxy=region_detections.xyxy,
                model=EFFICIENT_SAM_MODEL,
                device=DEVICE
            )
        if len(region_detections) > 0:
            detections = sv.Detections.merge([detections, region_detections])

    Detections = detections
    output_image = annotate_image(
        input_image=input_image,
        detections=detections,
        categories=categories,
        with_label=with_label,
        with_confidence=with_confidence
    )
    det_num = statistics.get_total_num(detections)
    class_count = statistics.get_class_count(detections)
    total_seg_proportion = statistics.get_total_seg_proportion(detections, input_image)
    average_confidence = statistics.get_average_confidence(detections)
    det_frame = statistics.get_det_dataFrame(detections, input_image)

    return output_image, total_seg_proportion, average_confidence, det_num, class_count, det_frame


def refine_segmentation(
    input_image: Image.Image,
    output_image: Image.Image,
    points: [],
    total_seg_proportion,
    det_frame,
    with_label: bool = True,
    with_confidence: bool = True
):

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
        return output_image, total_seg_proportion, det_frame, False

    # 根据point做seg
    input_image_np = np.array(input_image)
    point_mask = inference_with_points(
        image=input_image_np,
        points=filtered_points,
        model=EFFICIENT_SAM_MODEL,
        device=DEVICE
    )

    # 限制point_mask在对应的检测框内
    for i, index in enumerate(filtered_indices):
        x1, y1, x2, y2 = map(int, detections.xyxy[index][:4])
        mask = point_mask[i]
        cropped_mask = np.zeros_like(mask)
        cropped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        point_mask[i] = cropped_mask

    # 加和更新mask
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
    Detections = detections
    total_seg_proportion = statistics.get_total_seg_proportion(detections, input_image)
    det_frame = statistics.get_det_dataFrame(detections, input_image)
    return output_image, total_seg_proportion, det_frame, True


def undo_refine():
    global Detections, history
    points = []
    history.undo()
    latest_entry = history.get_latest_entry()
    if latest_entry:
        output_image = latest_entry['output_image']
        total_seg_proportion = latest_entry['total_seg_proportion']
        average_confidence = latest_entry['average_confidence']
        det_num = latest_entry['det_num']
        class_count = latest_entry['class_count']
        det_frame = latest_entry['det_frame']
        Detections = latest_entry['detections']
        return points, output_image, total_seg_proportion, average_confidence, det_num, class_count, det_frame
    else:
        return points, None, None, None, None, None, None


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
    if input_image is None:
        return None, None, None, None, None, None, None
    start = timeit.default_timer()
    categories = process_categories(categories)
    # 单病害单模型
    # 未完成，策略：循环单模型检测所有病害，结果通过from_ultralytics封装Detections，将多Detections通过merge方法合并
    # results = YOLO_WORLD_MODEL[categories[0]].infer(input_image, confidence=confidence_threshold, iou=iou_threshold,
    #                                                 text=categories)
    # 多病害模型
    detections = YOLO_WORLD_MODEL.infer(
        input_image,
        confidence=confidence_threshold,
        iou=iou_threshold,
        text=categories,
        class_agnostic=with_class_agnostic_nms
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
    total_num = statistics.get_total_num(detections)
    class_count = statistics.get_class_count(detections)
    total_seg_proportion = statistics.get_total_seg_proportion(detections, input_image)
    average_confidence = statistics.get_average_confidence(detections)
    duration = statistics.format_time(end - start)
    det_frame = statistics.get_det_dataFrame(detections, input_image)
    global Detections, Categories, history
    Detections = detections
    Categories = categories
    history.clear()
    history.add_entry(output_image, total_seg_proportion, average_confidence, total_num, class_count, det_frame,
                      Detections)
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
            detections = YOLO_WORLD_MODEL.infer(
                frame,
                confidence=confidence_threshold,
                iou=iou_threshold,
                text=categories,
                class_agnostic=with_class_agnostic_nms
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
    Detections: sv.Detections = None
    Categories: List[str] = None
    history = History()
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
            refine_mode_component = gr.Radio(
                choices=['Point', 'Box'],
                label='Refine Mode',
                value='Point',
                scale=1,
                interactive=True
            )
            with gr.Row():
                with gr.Column():
                    refine_button_component = gr.Button(
                        value='Refine',
                        scale=1,
                        variant='secondary'
                    )
                with gr.Column():
                    refine_undo_button_component = gr.Button(
                        value='Undo',
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
        inputs=[output_image_component, global_points, refine_mode_component, with_segmentation_component],
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
    refine_undo_button_component.click(
        fn=undo_refine,
        outputs=[
            global_points,
            output_image_component,
            total_seg_proportion_component,
            average_confidence_component,
            det_num_component,
            class_count_component,
            det_sheet_component
        ]
    )
    refine_button_component.click(
        fn=refine,
        inputs=[
            input_image_component,
            output_image_component,
            confidence_threshold_component,
            iou_threshold_component,
            global_points,
            refine_mode_component,
            total_seg_proportion_component,
            average_confidence_component,
            det_num_component,
            class_count_component,
            det_sheet_component,
            with_segmentation_component,
            with_label_component,
            with_confidence_component,
            with_class_agnostic_nms_component
        ],
        outputs=[
            output_image_component,
            global_points,
            total_seg_proportion_component,
            average_confidence_component,
            det_num_component,
            class_count_component,
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
