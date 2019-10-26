import collections
import numpy as np
import tensorflow as tf
from os import path
import rospy
import rospkg
import sys

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetector():
    def __init__(self, MODEL_NAME="default", MODELS_DIR=False):
        # Name of the directory containing the object detection module we're using
        self.MODEL_NAME = MODEL_NAME

        self.MODELS_DIR = MODELS_DIR
        if not MODELS_DIR:
            rospack = rospkg.RosPack()
            self.MODELS_DIR = path.join(rospack.get_path('humanoid_vision'), 'models')

        self.PATH_TO_CKPT = path.join(self.MODELS_DIR, self.MODEL_NAME, 'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = path.join(self.MODELS_DIR, self.MODEL_NAME, 'label_map.pbtxt')
        self.NUM_CLASSES = open(self.PATH_TO_LABELS, 'r').read().count('item')

        ## Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS)
        # label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        # self.category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def detect(self, frame):
        return self.sess.run([
            self.detection_boxes,
            self.detection_scores,
            self.detection_classes,
            self.num_detections
        ], feed_dict={self.image_tensor: frame})

    def visualize_detection(self, frame, detection, thresh=0.6):
        (boxes, scores, classes, num) = detection
        return vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=thresh)

    def detect_and_visualize(self, frame, thresh=0.6):
        detection = self.detect(frame)
        img = self.visualize_detection(frame[0], detection, thresh)
        return img, detection

    def getDetectionsCords(self, frame, thresh=0.6):
      pass
        # coordinates = vis_util.return_coordinates(
        #     frame,
        #     np.squeeze(self.boxes),
        #     np.squeeze(self.classes).astype(np.int32),
        #     np.squeeze(self.scores),
        #     self.category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=8,
        #     min_score_thresh=thresh)
        # return coordinates


# def return_coordinates(
#         image,
#         boxes,
#         classes,
#         scores,
#         category_index,
#         instance_masks=None,
#         instance_boundaries=None,
#         keypoints=None,
#         use_normalized_coordinates=False,
#         max_boxes_to_draw=20,
#         min_score_thresh=.5,
#         agnostic_mode=False,
#         line_thickness=4,
#         groundtruth_box_visualization_color='black',
#         skip_scores=False,
#         skip_labels=False):
#   # Create a display string (and color) for every box location, group any boxes
#   # that correspond to the same location.
#   box_to_display_str_map = collections.defaultdict(list)
#   box_to_color_map = collections.defaultdict(str)
#   box_to_instance_masks_map = {}
#   box_to_instance_boundaries_map = {}
#   box_to_score_map = {}
#   box_to_keypoints_map = collections.defaultdict(list)
#   if not max_boxes_to_draw:
#     max_boxes_to_draw = boxes.shape[0]
#   for i in range(min(max_boxes_to_draw, boxes.shape[0])):
#     if scores is None or scores[i] > min_score_thresh:
#       box = tuple(boxes[i].tolist())
#       if instance_masks is not None:
#         box_to_instance_masks_map[box] = instance_masks[i]
#       if instance_boundaries is not None:
#         box_to_instance_boundaries_map[box] = instance_boundaries[i]
#       if keypoints is not None:
#         box_to_keypoints_map[box].extend(keypoints[i])
#       if scores is None:
#         box_to_color_map[box] = groundtruth_box_visualization_color
#       else:
#         display_str = ''
#         if not skip_labels:
#           if not agnostic_mode:
#             if classes[i] in category_index.keys():
#               class_name = category_index[classes[i]]['name']
#             else:
#               class_name = 'N/A'
#             display_str = str(class_name)
#         if not skip_scores:
#           if not display_str:
#             display_str = '{}%'.format(int(100*scores[i]))
#           else:
#             display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
#         box_to_display_str_map[box].append(display_str)
#         box_to_score_map[box] = scores[i]
#         if agnostic_mode:
#           box_to_color_map[box] = 'DarkOrange'
#         else:
#           box_to_color_map[box] = STANDARD_COLORS[
#               classes[i] % len(STANDARD_COLORS)]

#   # Draw all boxes onto image.
#   coordinates_list = []
#   counter_for = 0
#   for box, color in box_to_color_map.items():
#     ymin, xmin, ymax, xmax = box
#     height, width, channels = image.shape
#     ymin = int(ymin*height)
#     ymax = int(ymax*height)
#     xmin = int(xmin*width)
#     xmax = int(xmax*width)
#     coordinates_list.append(
#         [display_str, ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)])
#     counter_for = counter_for + 1

#   return coordinates_list
