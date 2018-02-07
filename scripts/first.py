#!/usr/bin/env python


# license removed for brevity
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)

def talker():
    rospy.init_node('pythotest', anonymous=True)
    pub = rospy.Publisher('/test', Twist, queue_size=10)
    rospy.Subscriber("chatter", String, callback)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        twi = Twist()
        twi.linear.x = 0
        twi.angular.z = 90*3.14/180.0
        pub.publish(twi)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

