# A Pupil Plugin to Perceive the Object in Focus
- **This fork includes a real-time object-detection plugin that integrates eye-tracking (fixation data) to identify which object the user is currently fixated upon.  An example of it's potential is shown in controlling grasps of a robotic prosthetic hand.**
- Video Demo of Plugin: https://youtu.be/MqqQnCbzryA
- Video Demo of using the plugin to control a prosthetic hand: https://youtu.be/KYcfLEvbxSc
- The plugin is located in pupil_src/shared_modules/object_detector_app and is registered in world.py

![demo_example.png](/demo_example.png)
## This plugin, when run, shows the following real-time visualizations:
1. Bounding boxes & labels around recognized objects
2. Recent gaze points and detected fixations
3. The object on which the user is focused (a 'X' is placed at the center of the object closest to the user's recent gaze)

## Using the Object Detection Plugin to Control a Prosthetic Hand!
- This shows the incredible potential of combining real-time object detection with eye tracking.
- With an Arduino Mega 2560 and an OpenBionics 3D printed hand (along with the linear actuators for each finger), a grasp will be performed as soon an object is detected (so long as there is a predetermined grip associated with that object stored inside the Arduino firmware).
	- Within /robotic_hand_prosthetic_demo, you will find the Arduino firmware in /obj_rec_hand_demo and a ROS package /python3_receiver that contains a node 'talker.py.' This node receives information via zmq from the object detection plugin and publishes the label info to a rostopic called /object_detection_label.  There is a subscriber in the Arduino firmware for this same topic. 
- If you want to replicate this project or build upon it, I can provide a detailed how-to.  Feel free to reach out!

### Requirements:
- Tensorflow (1.2 or above)
- OpenCV 3.0 (or above)

### Other notes:
- There is functionality in the plugin to send object detection information via ZMQ. 
- This was developed using Tensorflow and was heavily inspired by Dat Tran's work (https://github.com/datitran/object_detector_app).
- _This was run solely on CPU, so the model (trained on the COCO dataset) had to be very lightweight. I was able to achieve about 10-15 FPS, at the cost of apples being perceived as donuts._



## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community! See the docs for more info on the [license](http://docs.pupil-labs.com/#license "License"). For support and custom licencing [contact us!](https://docs.pupil-labs.com/#email "email us")
