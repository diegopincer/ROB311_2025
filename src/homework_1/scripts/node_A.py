#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

# Global counter to track how many times the message comes back to node A
return_count = 0

def callback(msg):
    global return_count
    rospy.loginfo("Node A received: %s", msg.data)
    
    if return_count < 2:  # Message hasn't come back 3 times yet
        return_count += 1
        rospy.loginfo("Returning message for the %d time", return_count)
        pub.publish(msg.data + "A")  # Add "A" and publish again
    else:
        rospy.loginfo("Message received for the third time. Stopping...")
        rospy.signal_shutdown("Third reception completed")  # Stop after 3rd time

# Initialize node A
rospy.init_node('node_A')

# Publishes on the topic 'topic_B'
pub = rospy.Publisher('topic_B', String, queue_size=10)

# Subscribes to the topic 'topic_D'
sub = rospy.Subscriber('topic_A', String, callback)

# Publish initial message
rospy.sleep(1)  # Wait for everything to initialize
pub.publish("teste")

# Keeps the node running
rospy.spin()



