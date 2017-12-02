'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

'''
Adapted from Dat Tran's object detector app (https://github.com/datitran/object_detector_app)
and adapted for a pupil plugin by Jesse Weisberg.

Detects object but also places an 'X' in the center of the object at which you are gazing.

To be created: add fixation capability, use 
'''

from plugin import Visualizer_Plugin_Base
from pyglui.cygl.utils import draw_points_norm, draw_points, draw_polyline_norm, draw_rounded_rect, draw_x, RGBA
from pyglui import ui
from pyglui.pyfontstash import fontstash
from methods import denormalize
from multiprocessing import Process, Queue, Pool
from threading import Thread
from OpenGL.GL import *
from pyglui.ui import get_opensans_font_path
from glfw import *

import gl_utils
import os
import cv2
import sys
import time
import math
import multiprocessing
import numpy as np
import tensorflow as tf
import zmq
from msgpack import loads

from . utils.app_utils import draw_boxes_and_labels
from . object_detection.utils import label_map_util
from . object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join('/home/jesse/dev/pupil/pupil_src/shared_modules/object_detector_app/object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/jesse/dev/pupil/pupil_src/shared_modules/object_detector_app/object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class Object_Detection(Visualizer_Plugin_Base):
    """
    Object Detection allows you to detect objects in the world frame in real-time
    """
    uniqueness = "by_class"
    icon_chr = chr(0xe061)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, num_workers = 2, queue_size = 2, frame = None, run=False):
        super().__init__(g_pool)
        
        self.order = 1
        self.pupil_display_list = []
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.menu = None
        self.label = 'label'
        self.run = run

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', get_opensans_font_path())
        self.glfont.set_size(32)
        self.frame = frame

        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)
        self.input_q = Queue(maxsize=self.queue_size)
        self.output_q = Queue(maxsize=self.queue_size)
        for i in range(1):
            t = Thread(target=self.worker, args=(self.input_q, self.output_q))
            t.daemon = True
            t.start()

            #start another thread to publish information using zmq
            t2 = Thread(target=self.publish_detected_object)
            t2.daemon = True
            t2.start()


    def recent_events(self, events):

        for pt in events.get('gaze_positions',[]):
            self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']*0.8))
        self.pupil_display_list[:-1] = []

        if self.run == True:
            if 'frame' in events:
                frame = events['frame']
                img = frame.img
                self.frame = img           
                self.input_q.put(img)


    def gl_display(self):
        dist = self.pupil_display_list

        if self.output_q.empty():
            pass
        else:
            data = self.output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']

            height, width, _= self.frame.shape
            dist_list = []

            for point, name, color in zip(rec_points, class_names, class_colors):
                #define vertices necessary for drawing bounding boxes
                bottom_left, bottom_right = [point['xmin'], 1-point['ymax']], [point['xmax'], 1-point['ymax']]
                top_left, top_right = [point['xmin'],1-point['ymin']], [point['xmax'], 1-point['ymin']]
                top_left_label, top_right_label = top_left, [point['xmin']+len(name[0])*14/width, 1-point['ymin']]
                bottom_left_label, bottom_right_label = [point['xmin'],1-(point['ymin']+30/height)], [point['xmin']+len(name[0])*14/width, 1-(point['ymin']+30/height)]
                
                center_bb = [(point['xmax']+point['xmin'])/2.0, ((1-point['ymax'])+(1-point['ymin']))/2.0]

                #distance between gaze point and center of bounding box (center_bb)
                for pt,a in self.pupil_display_list:
                    dist = math.hypot(center_bb[0]-pt[0], center_bb[1]-pt[1])
                    # dist_list.append([dist, top_left[0]*width, (1-top_left[1])*height]) #for drawing text on detected object
                    dist_list.append([dist, [center_bb[0]*width, (1-center_bb[1])*height], name[0]])

                #draw bounding box, label box, write label
                verts_label = [top_left_label, top_right_label, bottom_right_label, bottom_left_label, top_left_label]
                verts = [top_left, top_right, bottom_right, bottom_left, top_left]
                draw_polyline_norm(verts, thickness=5, color=RGBA(color[2]/255,color[1]/255,color[0]/255,1.0))
                draw_polyline_norm(verts_label, thickness=5, color=RGBA(color[2]/255,color[1]/255,color[0]/255,1.0))

                self.glfont.set_color_float((1.0,1.0,1.0, 1.0))
                self.glfont.draw_text(top_left[0]*width, (1-top_left[1])*height, name[0])       
                self.glfont.set_align_string(v_align='left', h_align='top')

             
            #draw an x on the detected_object closest to the gaze       
            if dist_list:    
                draw_x([sorted(dist_list)[0][1]], size=50, thickness=5, color=RGBA(color[2]/255,color[1]/255,color[0]/255,.5))
                self.label = sorted(dist_list)[0][2]
                #print('self.label: ', self.label)
            #self.glfont.set_color_float((1.0,1.0,1.0, 1.0))
            #self.glfont.draw_text(sorted(dist_list)[0][1]+50, sorted(dist_list)[0][2], 'FIXATED OBJECT')
            #self.glfont.set_align_string(v_align='left', h_align='top')


    def detect_objects(self, image_np, sess, detection_graph):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        rect_points, class_names, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            category_index=category_index,
            min_score_thresh=.5
        )
        return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

    def worker(self, input_q, output_q):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        while True:
            frame = input_q.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_q.put(self.detect_objects(frame_rgb, sess, detection_graph))

        sess.close()

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Object Detection'
        self.menu.append(ui.Switch('run', self, label='Run Object Detection'))

    def deinit_ui(self):
        self.remove_menu()

    def publish_detected_object(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        addr = '127.0.0.1'  # remote ip or localhost
        port = "5556"  # same as in the pupil remote gui
        socket.bind("tcp://{}:{}".format(addr, port))
        time.sleep(1)
        while self.label is not None:
            topic = 'detected_object'
            #print ('%s %s' % (topic, self.label))
            try:
                socket.send_string('%s %s' % (topic, self.label))
            except TypeError:
                socket.send('%s %s' % (topic, self.label))