#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo("Node D received: %s", msg.data)
    # Adds "D" to the message and publishes it
    pub.publish(msg.data + "D")  

# Initialize node D
rospy.init_node('node_D')

# Publishes on the topic 'topic_A'
pub = rospy.Publisher('topic_A', String, queue_size=10)

# Subscribes to the topic 'topic_D'
sub = rospy.Subscriber('topic_D', String, callback)

# Keeps the node running
rospy.spin()

