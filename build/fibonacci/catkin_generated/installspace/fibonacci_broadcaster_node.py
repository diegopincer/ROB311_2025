#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32

def fibonacci():
    pub = rospy.Publisher('fibonacci', Int32, queue_size=10)
    rospy.init_node('fibonacci_broadcaster', anonymous=True)
    rate = rospy.Rate(1)  # 1 Hz

    a, b = 0, 1
    while not rospy.is_shutdown():
        rospy.loginfo(a)
        pub.publish(a)
        a, b = b, a + b
        rate.sleep()

if __name__ == '__main__':
    try:
        fibonacci()
    except rospy.ROSInterruptException:
        pass
