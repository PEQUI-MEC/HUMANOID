#!/usr/bin/env python
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray

from humanoid_vision import ObjectDetector

frame = None
vis_pub = rospy.Publisher('/PMH/object_detection', Image)
detection_pub = rospy.Publisher('/vision/object_detection', Float32MultiArray)
cv_bridge = CvBridge()
threshold = 0.9

def publish_ball_angle(frame):
    height, width, dim = frame.shape
    frame_expanded = np.expand_dims(frame, axis=0)
    #(boxes, scores, classes, num) = detector.detect(frame_expanded)
    img, detection = detector.detect_and_visualize(frame_expanded, thresh=threshold)
    (boxes, scores, classes, num) = detection
    # publish ball angle
    best_score_i = np.argmax(scores[0])
    if(scores[0][best_score_i]>threshold):
        (ymin, xmin, ymax, xmax) = boxes[0][best_score_i]
        msg = Float32MultiArray()
        msg.data = [ymin, xmin, ymax, xmax]
        detection_pub.publish(msg)
        
    vis_pub.publish(cv_bridge.cv2_to_imgmsg(img))

def frame_callback(msg):
    global frame
    frame = cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    #rospy.loginfo(frame.shape)

if __name__ == '__main__':
    rospy.init_node('ball_detection')
    rospy.loginfo('Pequi Mecanico Humanoid - Ball Detection Node')
    rate = rospy.Rate(100)

    detector = ObjectDetector(MODELS_DIR=rospy.get_param('~models_dir', False))

    rospy.loginfo('Finished loading! Starting detection...')

    frame_sub = rospy.Subscriber('/PMH/camera_frame', Image, frame_callback)

    while not rospy.is_shutdown():
        rate.sleep()

        if frame is not None:
            publish_ball_angle(frame)
            frame = None
