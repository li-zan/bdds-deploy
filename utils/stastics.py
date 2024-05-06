import supervision as sv
import pandas as pd


def get_total_num(detections: sv.Detections):
    return len(detections)


def get_class_count(detections: sv.Detections):
    class_list = list(detections.data['class_name'])
    class_dict = {}
    if len(class_list) == 0:
        class_dict['None'] = [0]
        return pd.DataFrame(class_dict)
    for i in class_list:
        if i in class_dict:
            class_dict[i][0] += 1
        else:
            class_dict[i] = [1]
    return pd.DataFrame(class_dict)


def format_time(time):
    return f"{time:.3f} s"


def get_average_confidence(detections: sv.Detections):
    if len(detections) == 0:
        return '0.00'
    average_confidence = sum(detections.confidence) / len(detections)
    return f'{average_confidence:.2f}'


def get_total_seg_proportion(detections: sv.Detections):
    if len(detections) == 0:
        return '0.00 %'
    total_seg_area = sum(detections.area)
    total_area = detections.mask[0].shape[0] * detections.mask[0].shape[1]
    return f'{total_seg_area / total_area * 100:.2f} %'


def get_det_dataFrame(detections: sv.Detections):
    if len(detections) == 0:
        det_dict = {
            "id": ['None'],
            "class_name": ['None'],
            "confidence": ['None'],
            "area": ['None'],
            "upper_left": ['None'],
            "lower_right": ['None']
        }
        return pd.DataFrame(det_dict)

    total_area = detections.mask[0].shape[0] * detections.mask[0].shape[1]

    det_dict = {
        "id": [
            idx
            for idx in range(1, len(detections.confidence)+1)
        ],
        "class_name": list(detections.data['class_name']),
        "confidence": [f"{x:.2f}" for x in detections.confidence],
        "area": [f"{x/total_area*100:.2f} %" for x in detections.area],
        "upper_left": [
            (int(x), int(y))
            for x, y, _, _ in detections.xyxy.astype(int)
        ],
        "lower_right": [
            (int(x), int(y))
            for _, _, x, y in detections.xyxy.astype(int)
        ]
    }
    return pd.DataFrame(det_dict)
