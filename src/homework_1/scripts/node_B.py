#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo("Node B received: %s", msg.data)
    pub.publish(msg.data + "B")  # Adds "B" to the message

# Initialize node B
rospy.init_node('node_B')

# Publishes on the topic 'topic_C'
pub = rospy.Publisher('topic_C', String, queue_size=10)

# Subscribes to the topic 'topic_B'
sub = rospy.Subscriber('topic_B', String, callback)

# Keeps the node running
rospy.spin()

