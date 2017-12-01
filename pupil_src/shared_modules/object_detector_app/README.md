# Object-Detector-App for Pupil 

A real-time object recognition application in your world frame using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).  The idea here is to integrate eye-tracking with object recognition, so objects detected are displayed in your world frame and an 'X' overlays the object closest to your gaze.

## Getting Started
1.  After starting 'main.py' open the 'Object Detection' Plugin from the menu and click 'Run Object Detection'.

## Tests
```
pytest -vs utils/
```

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

## Notes
- OpenCV 3.1 might crash on OSX after a while, so that's why I had to switch to version 3.0. See open issue and solution [here](https://github.com/opencv/opencv/issues/5874).
- Moving the `.read()` part of the video stream in a multiple child processes did not work. However, it was possible to move it to a separate thread.

## Copyright for original object_detector_app
See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Dat Tran](http://www.dat-tran.com/).
