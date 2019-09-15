#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray

data = [0., 0., 0.]

if __name__ == '__main__':
	cmd_pub = rospy.Publisher('PMH/vision_status', Float32MultiArray, queue_size=1)
	rospy.init_node('manual_vision', anonymous=True)

	while not rospy.is_shutdown():
		data[0] = float(input("Angulo vertical (-180, 180): "))
		data[1] = float(input("Angulo horizontal (-180, 180): "))
		data[2] = float(input("Esta com a bola? "))

		msg = Float32MultiArray()
		msg.data = data
		cmd_pub.publish(msg)
		print()
