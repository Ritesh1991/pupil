#!/usr/bin/env python
#!/usr/bin/env python3

'''
Subscribes to a zmq socket and publishes that information to a ros topic.  This is one workaround for using
Python 2 and Python 3 in the same ROS application.

In my case, this receives real-time object detection info from a script in Python 3 and publishes to a rostopic.

Author: Jesse Weisberg
'''
import rospy
from std_msgs.msg import String
import sys
import zmq
from msgpack import loads
import time
import pyttsx
from datetime import datetime 
from espeak import espeak

fixated_object_label = None

#subscribe to detected object from object_detection_pupil.py (Pupil object detection plugin) via zmq
def subscribe_detected_object():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    addr = '127.0.0.1'  # remote ip or localhost
    port = "5556"  # same as in the pupil remote gui
    print('retrieving objects...')
    socket.connect("tcp://{}:{}".format(addr, port))

    #subscribe to detected_objects topic
    while True:
        try:
            socket.setsockopt_string(zmq.SUBSCRIBE, 'detected_object')
        except TypeError:
            socket.setsockopt(zmq.SUBSCRIBE, 'detected_object')
        #process object
        detected_object = socket.recv_string() 
        if len(detected_object.split())==3:
            fixated_object_label = detected_object.split()[1]
            confidence = detected_object.split()[2]
        if len(detected_object.split())==4:
            fixated_object_label = detected_object.split()[1] + ' ' + detected_object.split()[2]
            confidence = detected_object.split()[3]

        # Potential improvement idea with emg sensory feedback
        # activate grasp for robotic manipulator: turn on "ready to execute switch"
        # time.sleep(3), during this time wait for emg sensory input
        # set up another rostopic that with emg sensory input, 
        # arduino reads that if higher than thresh, execute predetermined motion planning/grasp  
        return fixated_object_label


# publish detected object to a ros topic
def publish_detected_object():
    pub = rospy.Publisher('object_detection_label', String, queue_size=10)
    rospy.init_node('detected_objects', anonymous=True)
    rate = rospy.Rate(20) # 20hz

    while not rospy.is_shutdown():
        fixated_object_label = subscribe_detected_object()
        rospy.loginfo(fixated_object_label)
        pub.publish(fixated_object_label)

        espeak.synth(fixated_object_label)
        while espeak.is_playing():
             pass

        rospy.sleep(3)
        rate.sleep()
    

if __name__ == '__main__':
    try:
        publish_detected_object()
    except rospy.ROSInterruptException:
        pass
