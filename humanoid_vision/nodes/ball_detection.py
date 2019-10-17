#!/usr/bin/env python
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray

from humanoid_vision.object_detection import ObjectDetector

gimbal_pos = np.zeros((2,), dtype="float32")
frame = None
vis_pub = rospy.Publisher('/PMG/object_detection', Image)
gimbal_pub = rospy.Publisher('/Bioloid/gimbal_cmd', Int16MultiArray)
cv_bridge = CvBridge()
threshold = 0.6

kp = 1

def publish_ball_angle(frame):
    gimbal_pos_frame = gimbal_pos.copy()
    height, width, dim = frame.shape
    frame_expanded = np.expand_dims(frame, axis=0)
    #(boxes, scores, classes, num) = detector.detect(frame_expanded)
    img, detection = detector.detect_and_visualize(frame_expanded, thresh=threshold)
    (boxes, scores, classes, num) = detection
    # publish ball angle
    best_score_i = np.argmax(scores[0])
    if(scores[0][best_score_i]>threshold):
        (ymin, xmin, ymax, xmax) = boxes[0][best_score_i]
        centroid = [((xmax+xmin)/2)-0.5, ((ymax+ymin)/2)-0.5]
        centroid[0] *= 450*kp/2
        centroid[1] *= -350*kp/2

        msg = Int16MultiArray()
        msg.data = [int(gimbal_pos_frame[0]*10 + centroid[0]), int(gimbal_pos_frame[1]*10 + centroid[1])]
        print(msg.data)
        gimbal_pub.publish(msg)
    vis_pub.publish(cv_bridge.cv2_to_imgmsg(img))

def frame_callback(msg):
    global frame
    frame = cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    #rospy.loginfo(frame.shape)

def gimbal_callback(msg):
    gimbal_pos[0] = msg.data[0]
    gimbal_pos[1] = msg.data[1]


if __name__ == '__main__':
    rospy.init_node('ball_detection')
    rospy.loginfo('Pequi Mecanico Humanoid - Ball Detection Node')
    rate = rospy.Rate(4)

    detector = ObjectDetector(MODELS_DIR=rospy.get_param('~models_dir', False))

    rospy.loginfo('Finished loading! Starting detection...')

    gimbal_sub = rospy.Subscriber('/Bioloid/gimbal_pos', Float32MultiArray, gimbal_callback)
    frame_sub = rospy.Subscriber('/PMH/camera_frame', Image, frame_callback)

    msg = Int16MultiArray()
    msg.data = [0, 0]
    gimbal_pub.publish(msg)

    while not rospy.is_shutdown():
        rate.sleep()

        if frame is not None:
            publish_ball_angle(frame)
            frame = None
