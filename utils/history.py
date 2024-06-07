import copy
from PIL import Image


class History:
    def __init__(self):
        self.history = []

    def add_entry(self, output_image: Image.Image, total_seg_proportion, average_confidence,
                  det_num, class_count, det_frame, detections):
        entry = {
            'output_image': output_image.copy(),
            'total_seg_proportion': total_seg_proportion,
            'average_confidence': average_confidence,
            'det_num': det_num,
            'class_count': class_count,
            'det_frame': det_frame,
            'detections': copy.deepcopy(detections)
        }
        self.history.append(entry)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()

    def get_latest_entry(self):
        if len(self.history) > 0:
            return copy.deepcopy(self.history[-1])
        else:
            print("No entries in history")
            return None

    def clear(self):
        self.history = []
