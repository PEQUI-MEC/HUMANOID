#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray
import threading

from humanoid_control.control import Control

def publish_joint_pos():
  data = to_deg(control.msg_to_micro).astype(np.int16)
  msg = Int16MultiArray()
  msg.data = data
  joint_pub.publish(msg)

def to_deg(angles):
  return np.array(angles) * 1800 / np.pi

if __name__ == "__main__":
  rospy.init_node('control_node')
  rospy.loginfo('Pequi Mecanico Humanoid - Control Node')
  rate = rospy.Rate(120)

  control = Control(gravity_compensation_enable=True)

  joint_pub = rospy.Publisher('/Bioloid/joint_pos', Int16MultiArray, queue_size=1)
  rospy.Subscriber('Bioloid/visao_cmd', Float32MultiArray, control.visao_cmd_callback)

  control_thread = threading.Thread(target=control.run)
  control_thread.daemon = True
  rospy.loginfo('Starting control thread...')
  control_thread.start()

  try:
    while not rospy.is_shutdown():
      publish_joint_pos()
      rate.sleep()
  except KeyboardInterrupt:
    pass

  rospy.loginfo('Stopping control thread...')
  control.running = False
  control_thread.join()
