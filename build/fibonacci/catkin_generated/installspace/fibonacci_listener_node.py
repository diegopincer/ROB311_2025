

import rospy
from std_msgs.msg import Int32

def callback(data):
    rospy.loginfo("I heard %d", data.data)

def listener():
    rospy.init_node('fibonacci_listener', anonymous=True)
    rospy.Subscriber('fibonacci', Int32, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

