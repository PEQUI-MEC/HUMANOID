#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray, String
import threading

from humanoid_control.control import Control
from humanoid_control.utils import rad_to_deg

def publish_joint_pos():
  data = rad_to_deg(control.angulos[:]).astype(np.int16)
  msg = Int16MultiArray()
  msg.data = data
  joint_pub.publish(msg)

def command_callback(msg):
  cmd = msg.data
  rospy.loginfo('Received command: ' + cmd)
  if cmd == 'reset':
    control.reset()
  elif cmd == 'walk':
    if control.manual_mode:
      control.visao_bola = True
  elif cmd == 'ready':
    control.enable = True
  elif cmd == 'set_mode_manual':
    control.manual_mode = True
  elif cmd == 'set_mode_auto':
    control.manual_mode = False

if __name__ == "__main__":
  rospy.init_node('control_node')
  rospy.loginfo('Pequi Mecanico Humanoid - Control Node')
  rate = rospy.Rate(120)

  control = Control(gravity_compensation_enable=False)

  joint_pub = rospy.Publisher('/PMH/joint_pos', Int16MultiArray, queue_size=1)
  rospy.Subscriber('PMH/vision_status', Float32MultiArray, control.vision_status_callback)
  rospy.Subscriber('PMH/control_command', String, command_callback)

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
