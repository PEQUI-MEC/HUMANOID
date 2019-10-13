#!/usr/bin/env python
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image

from humanoid_vision.object_detection import ObjectDetector

frame = False


def publish_ball_angle(frame):
    frame_expanded = np.expand_dims(frame)
    (boxes, scores, classes, num) = detector.detect(frame_expanded)
    # publish ball angle


def frame_callback(msg):
    frame = cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    rospy.loginfo(frame.shape)


if __name__ == '__main__':
    rospy.init_node('ball_detection')
    rospy.loginfo('Pequi Mecanico Humanoid - Ball Detection Node')
    rate = rospy.Rate(4)

    cv_bridge = CvBridge()
    detector = ObjectDetector(MODELS_DIR=rospy.get_param('~models_dir', False))

    rospy.loginfo('Finished loading! Starting detection...')

    frame_sub = rospy.Subscriber('/PMH/camera_frame', Image, frame_callback)

    while not rospy.is_shutdown():
        rate.sleep()

        if frame:
            publish_ball_angle(frame)
            frame = False
