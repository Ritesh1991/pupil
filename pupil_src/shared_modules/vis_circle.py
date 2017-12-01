'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from player_methods import transparent_circle
from plugin import Visualizer_Plugin_Base
from pyglui import ui
from pyglui.cygl.utils import draw_points_norm,RGBA
from methods import denormalize


class Vis_Circle(Visualizer_Plugin_Base):
    """
    Real-time object detection and classification (using darkflow/YOLO) overlayed on world image.
    """
    uniqueness = "not_unique"
    icon_chr = chr(0xe061)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool,radius=20,color=(0.0,0.7,0.25,0.2),thickness=2,fill=True, min_confidence=.2):
        super().__init__(g_pool)
        self.order = .9

        # initialize empty menu
        self.menu = None

        self.min_confidence = min_confidence
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]
        self.radius = radius
        self.thickness = thickness
        self.fill = fill
        self.pupil_display_list = []

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        if self.fill:
            thickness = -1
        else:
            thickness = self.thickness
        
        for pt in events.get('gaze_positions',[]):
            self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']*0.8))
        self.pupil_display_list[:-3] = []

        # pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('gaze_positions',[]) if pt['confidence']>=self.min_confidence]
        for pt, a in self.pupil_display_list:
            #print('vis_circle', pt, a)
            transparent_circle(frame.img, pt, radius=self.radius, color=(self.b, self.g, self.r, self.a), thickness=thickness)
            draw_points_norm([pt],
                        size=self.radius,
                        color=RGBA(self.r,self.g,self.b,self.a))
            

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Gaze Circle'
        self.menu.append(ui.Slider('radius',self,min=1,step=1,max=100,label='Radius'))
        self.menu.append(ui.Slider('thickness',self,min=1,step=1,max=15,label='Stroke width'))
        self.menu.append(ui.Switch('fill',self,label='Fill'))

        color_menu = ui.Growing_Menu('Color')
        color_menu.collapsed = True
        color_menu.append(ui.Info_Text('Set RGB color components and alpha (opacity) values.'))
        color_menu.append(ui.Slider('r',self,min=0.0,step=0.05,max=1.0,label='Red'))
        color_menu.append(ui.Slider('g',self,min=0.0,step=0.05,max=1.0,label='Green'))
        color_menu.append(ui.Slider('b',self,min=0.0,step=0.05,max=1.0,label='Blue'))
        color_menu.append(ui.Slider('a',self,min=0.0,step=0.05,max=1.0,label='Alpha'))
        self.menu.append(color_menu)

    def deinit_ui(self):
        self.remove_menu()

    def get_init_dict(self):
        return {'radius':self.radius,'color':(self.r, self.g, self.b, self.a),'thickness':self.thickness,'fill':self.fill}



    # fixated_object_label = None

# def publish_detected_object(label):
#     context = zmq.Context()
#     socket = context.socket(zmq.PUB)
#     addr = '127.0.0.1'  # remote ip or localhost
#     port = "5556"  # same as in the pupil remote gui
#     socket.bind("tcp://{}:{}".format(addr, port))
#     time.sleep(1)
#     while label is not None:
#         topic = 'detected_object'
#         #print ('%s %s' % (topic, label))
#         try:
#             socket.send_string('%s %s' % (topic, label))
#         except TypeError:
#             socket.send('%s %s' % (topic, label))
#         break
      

# def detect_gazed_object():
#     # Run on yolo trained on COCO dataset (COCO is a dataset comprised of common household objects)
#     options = {"model": "cfg/yolo.cfg", "load": "weights/yolo.weights", "demo": 'camera', "threshold": 0.2}

#     # Run on tiny yolo (for ultimate speed)
#     #options = {"model": "cfg/tiny-yolo.cfg", "load": "weights/tiny-yolo.weights", "threshold": 0.2}
#     tfnet = TFNet(options)

#     # Predetermined radial distortion coefficients and intrinsic camera parameters for undistorting later on.
#     dist_coefs = np.array(pre_recorded_calibrations['Pupil Cam1 ID2']['(1280, 720)']['dist_coefs'])
#     camera_matrix = np.array(pre_recorded_calibrations['Pupil Cam1 ID2']['(1280, 720)']['camera_matrix'])

#     frame_width = 1280  # Alternate potential dimensions: 320x240
#     frame_height = 720
#     cam = cv2.VideoCapture(-1) # world camera: index=2, eye camera: index=3
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
#     while(True):
#         # Capture frame by frame
#         _, frame = cam.read()

#         # Perform radial undistortion (Pupil world camera has significant radial distortion which adversely affects detection/classification).
#         frame = cv2.undistort(frame, camera_matrix, dist_coefs)
#         cv2.imshow('Object Detection',frame)

#         # Uses neural network (darkflow) to predict detected objects and respective classifications. 
#         objects_detected = tfnet.return_predict2(frame)
#         #print(objects_detected)

#         # Collect normalized gaze_data, denormalize according to camera resolution (bottom-left corner is (0,0), top-right is (1,1)).
#         gaze_data = getGazeData()
#         gaze_x = (gaze_data['gaze_coord'][0])*frame_width
#         gaze_y = (1-gaze_data['gaze_coord'][1])*frame_height  # Y-coordinate was flipped initially

#         # if gaze_data['confidence']>.7:
#             # print(gazeX, gazeY, ' confidence:', gaze_data['confidence'])

#         # Append gaze point to real-time stream as a green dot.
#         frame = cv2.circle(frame, (int(gaze_x), int(gaze_y)), 10, (0,255,0), -1)
#         cv2.imshow('Object Detection',frame)

#         if objects_detected:
#             # Weed out irrelevant bounding boxes (ones that aren't close to your gaze)
#             # Calculate 'radius' of each bounding box - if the gaze point is outside of this radius, don't show the box
#             for obj in objects_detected:
#                 radius = math.hypot(obj[2][1] - obj[2][0], obj[3][1] - obj[3][0])
#                 if radius > obj[4]:
#                     obj.append(radius) 
#                 else:
#                     objects_detected.remove(obj)

#             # Sort detected objects in increasing order by distance between center of object and gaze coordinate (on the image plane)
#             if objects_detected:  # If there are still any detected objects after 'weeding out' irrelevant objects.
#                 sortedClasses = sorted(objects_detected, key=itemgetter(4),  reverse=False)
#                 closest_obj = True;
#                 for obj in sortedClasses:
#                     top_left_x, top_left_y = obj[2][0], obj[2][1] 
#                     bottom_right_x, bottom_right_y = obj[3][0], obj[3][1]
#                     bounding_box_color = (255,0,0) # Default bounding box color is blue.
#                     if(closest_obj==True):
#                         global fixated_object_label
#                         fixated_object_label = obj[0]
#                         #print(fixated_object_label)
#                         p2 = Process(target = publish_detected_object(fixated_object_label))
#                         p2.start()
#                         # publish_detected_object(fixated_object_label) via zmq so we can use in ROS
#                         bounding_box_color = (0,255,0) # Closest object has a green bounding box.
#                         closest_obj = False
#                         p2.terminate()
#                     # Append bounding boxes along with classification labels
#                     frame = cv2.rectangle(frame,(top_left_x, top_left_y),(bottom_right_x,bottom_right_y),bounding_box_color,3)
#                     label_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#                     cv2.putText(frame, obj[0], (top_left_x,top_left_y-15), label_font, 1,(255,255,255),1,cv2.LINE_AA)
#                     cv2.imshow('Object Detection',frame)        
#             else:
#                 continue
