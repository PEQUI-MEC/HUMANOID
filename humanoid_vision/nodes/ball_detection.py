#!/usr/bin/env python
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray, String, Empty
from std_srvs.srv import Trigger, TriggerResponse
from humanoid_vision import ObjectDetector
import time
import cv2
import math

prev_gray = None
frame = None
vis_pub = rospy.Publisher('/PMH/object_detection', Image)
detection_pub = rospy.Publisher('/vision/object_detection', Float32MultiArray)
defend_pub = rospy.Publisher('/PMH/defend/result', String)
cv_bridge = CvBridge()
threshold = 0.9

"""
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
"""
defend_flag = False
detection_count = 0
detection_bbox = [0, 0, 0, 0]
init_p0 = None
p0 = None
color = np.random.randint(0,255,(100,3))
mask = None
good_new = None
dist = 0

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
        detection_count += 1
        detection_bbox = msg.data

    vis_pub.publish(cv_bridge.cv2_to_imgmsg(img))


def frame_callback(msg):
    global frame, detection_count, detection_bbox, prev_gray, p0, color, mask, good_new, dist, init_p0, defend_flag
    frame = cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    #rospy.loginfo(frame.shape)
    if(defend_flag):
        if(detection_count < 10):
            height, width, dim = frame.shape
            frame_expanded = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = detector.detect(frame_expanded)
            best_score_i = np.argmax(scores[0])
            if(scores[0][best_score_i]>threshold):
                (ymin, xmin, ymax, xmax) = boxes[0][best_score_i]
                detection_bbox = [int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)]
                #detection_pub.publish(msg)
                detection_count += 1
        else:
            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )
            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            point_max = (detection_bbox[3], detection_bbox[2])
            point_min = (detection_bbox[1], detection_bbox[0])
            crop_img = gray[point_min[1]:point_max[1], point_min[0]:point_max[0]]
            if(p0 is None):
                p0 = cv2.goodFeaturesToTrack(crop_img, mask = None, **feature_params)
                mask = np.zeros_like(frame)
                for i, x in enumerate(p0):
                    p0[i][0][0] = x[0][0]+(point_min[0])
                    p0[i][0][1] = x[0][1]+(point_min[1])
                init_p0 = p0.copy()
            else:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                if(p1 is None):
                    p0 = None
                    detection_count = 0
                    dist = 0
                    print("lost")
                    return
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                dist = 0
                for i,(new,old) in enumerate(zip(good_new,init_p0)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    dist += (math.sqrt((c-a)**2 + (d-b)**2)*(1 if c-a > 0 else -1))/len(good_new)
                frame = cv2.add(frame,mask)
                print(dist)
                if(abs(dist) > 70):
                    msg = String()
                    msg.data = "RIGHT" if dist < 0 else "LEFT"
                    defend_pub.publish(msg)
                    p0 = None
                    detection_count = 0
                    dist = 0
                    defend_flag = False
            """
            for i in p0:
                x,y = i.ravel()
                cv2.circle(frame,(x,y),3,255,-1)
            """
            #print(detection_bbox)
            #sprint(prev)
            img_rect = cv2.rectangle(frame, point_max, point_min, (255, 0, 0), 2)
            vis_pub.publish(cv_bridge.cv2_to_imgmsg(img_rect))
            if(good_new is not None):
                p0 = good_new.reshape(-1,1,2)
            prev_gray = gray.copy()

def defend_callback(msg):
    global defend_flag
    defend_flag = True

if __name__ == '__main__':
    rospy.init_node('ball_detection')
    rospy.loginfo('Pequi Mecanico Humanoid - Ball Detection Node')
    rate = rospy.Rate(100)

    detector = ObjectDetector(MODELS_DIR=rospy.get_param('~models_dir', False))

    rospy.loginfo('Finished loading! Starting detection...')

    frame_sub = rospy.Subscriber('/PMH/camera_frame', Image, frame_callback)
    defend_sub = rospy.Subscriber('/PMH/defend', Empty, defend_callback)
    rospy.spin()
    """
    while not rospy.is_shutdown():
        rate.sleep()

        if frame is not None:
            publish_ball_angle(frame)
            frame = None
    """