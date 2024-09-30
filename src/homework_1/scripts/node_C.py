#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo("Node C received: %s", msg.data)
    pub.publish(msg.data + "C")  # Adds "C" to the message

# Initialize node C
rospy.init_node('node_C')

# Publishes on the topic 'topic_D'
pub = rospy.Publisher('topic_D', String, queue_size=10)

# Subscribes to the topic 'topic_C'
sub = rospy.Subscriber('topic_C', String, callback)

# Keeps the node running
rospy.spin()

