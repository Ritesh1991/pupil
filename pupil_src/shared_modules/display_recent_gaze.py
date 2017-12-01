'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import System_Plugin_Base
from pyglui.cygl.utils import draw_points_norm,RGBA
from pyglui import ui
from methods import denormalize
from multiprocessing import Process 

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class Display_Recent_Gaze(System_Plugin_Base):
    """
    DisplayGaze shows the most
    recent gaze position on the screen
    """

    def __init__(self, g_pool, num_workers = 2, queue_size = 5):
        super().__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []
        self.num_workers = num_workers
        self.queue_size = queue_size

        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)

        input_q = Queue(maxsize=args.queue_size)
        output_q = Queue(maxsize=args.queue_size)
        pool = Pool(args.num_workers, worker, (input_q, output_q))


    def recent_events(self,events, image_np, sess, detection_graph):

        for pt in events.get('gaze_positions',[]):
            self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']*0.8))
        self.pupil_display_list[:-1] = []
        
        if 'frame' in events:

            frame = events['frame']
            frame = frame.img

             # Collect normalized gaze_data, denormalize according to camera resolution (bottom-left corner is (0,0), top-right is (1,1)).
            width = frame.shape[1]
            height = frame.shape[0] 

            input_q.put(frame)

            t=time.time()

            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', output_rgb)

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))



    def gl_display(self):
        for pt,a in self.pupil_display_list:
            #print('display recent:', pt, a)
            #This could be faster if there would be a method to also add multiple colors per point
            draw_points_norm([pt],
                        size=100,
                        color=RGBA(1.,.2,.4,a))

    def get_init_dict(self):
        return {}

    def detect_objects(image_np, sess, detection_graph):
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
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np

    def worker(input_q, output_q):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        fps = FPS().start()
        while True:
            fps.update()
            frame = input_q.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_q.put(detect_objects(frame_rgb, sess, detection_graph))

        fps.stop()
        sess.close()



    # def publish_detected_object(label):
    #         context = zmq.Context()
    #         socket = context.socket(zmq.PUB)
    #         addr = '127.0.0.1'  # remote ip or localhost
    #         port = "5556"  # same as in the pupil remote gui
    #         socket.bind("tcp://{}:{}".format(addr, port))
    #         time.sleep(1)
    #         while label is not None:
    #             topic = 'detected_object'
    #             #print ('%s %s' % (topic, label))
    #             try:
    #                 socket.send_string('%s %s' % (topic, label))
    #             except TypeError:
    #                 socket.send('%s %s' % (topic, label))
    #             break