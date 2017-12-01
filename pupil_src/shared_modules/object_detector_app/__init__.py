# import detector classes from sibling files
import sys
sys.path.insert(0, '/home/jesse/dev/pupil/pupil_src/shared_modules/object_detector_app')
#sys.path.insert(1, '/home/jesse/dev/pupil/pupil_src/shared_modules')
# print('2', sys.path)
from . utils import *
from . object_detection_pupil import Object_Detection
from . object_detection import *
#from . object_detection_multithreading import *