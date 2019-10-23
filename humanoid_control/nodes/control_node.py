#!/usr/bin/env python
from imu_bno055.msg import EulerAngles
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray, String, UInt8MultiArray
import threading

from humanoid_control.control import Control
from humanoid_control.utils import rad_to_deg

STATUS_RATE_DIVIDER = 24
DIVIDER_COUNTER = 0


def publish_joint_pos():
  data = rad_to_deg(control.angulos[:]).astype(np.int16)
  msg = Int16MultiArray()
  msg.data = data
  joint_pub.publish(msg)


def publish_control_status():
  msg = UInt8MultiArray()
  msg.data = [control.state_encoder[control.state], control.manual_mode]
  status_pub.publish(msg)


def publish_interpolation_pos():
  msg = Int16MultiArray()
  try:
    msg.data = np.array(next(control.interpolation), dtype=np.int16)
  except StopIteration:
    msg.data = [0]
    control.reset()
  interpolation_pub.publish(msg)


def command_callback(msg):
  cmd = msg.data
  if cmd == 'next_interpolation':
    publish_interpolation_pos()
  elif cmd == 'reset':
    # rospy.signal_shutdown('reset')
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


def imu_callback(msg):
  control.update_robot_orientation(msg.angles.x, msg.angles.y, msg.angles.z)


if __name__ == "__main__":
  rospy.init_node('control_node')
  rospy.loginfo('Pequi Mecanico Humanoid - Control Node')
  rate = rospy.Rate(110)

  control = Control(gravity_compensation_enable=True)

  joint_pub = rospy.Publisher('/PMH/joint_pos', Int16MultiArray, queue_size=1)
  status_pub = rospy.Publisher('/PMH/control_status', UInt8MultiArray, queue_size=1)
  interpolation_pub = rospy.Publisher('/PMH/interpolation_pos', Int16MultiArray, queue_size=1)
  rospy.Subscriber('PMH/vision_status', Float32MultiArray, control.vision_status_callback)
  rospy.Subscriber('PMH/control_command', String, command_callback)

  rospy.Subscriber('PMH/imu/euler_angles', EulerAngles, imu_callback)

  control_thread = threading.Thread(target=control.run)
  control_thread.daemon = True
  rospy.loginfo('Starting control thread...')
  control_thread.start()

  try:
    while not rospy.is_shutdown():
      rate.sleep()
      publish_joint_pos()

      DIVIDER_COUNTER += 1
      if DIVIDER_COUNTER >= STATUS_RATE_DIVIDER:
        publish_control_status()
        DIVIDER_COUNTER = 0
  except KeyboardInterrupt:
    pass

  rospy.loginfo('Stopping control thread...')
  control.running = False
  control_thread.join()
