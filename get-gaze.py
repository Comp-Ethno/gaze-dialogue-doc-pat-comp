import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


fps_time = 0
frames_per_second = []


two_ears = []
facing = []
nose_coord = []
left_eye_coord = []
right_eye_coord = []
frame_prediction = []


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def detect(fps_time):
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        parser.add_argument('--camera', type=str, default='./low-res-video/sample.mp4')

        parser.add_argument('--resize', type=str, default='0x0',
                            help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                            help='if provided, resize heatmaps before they are post-processed. default=1.0')

        parser.add_argument('--model', type=str, default='mobilenet_v2_large', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')
        
        parser.add_argument('--tensorrt', type=str, default="False",
                            help='for tensorrt process.')
        args = parser.parse_args()
        
        w, h = model_wh(args.resize)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
               
        cam = cv2.VideoCapture(args.camera)
        ret_val, image = cam.read()

        frames_per_second.append(cam.get(cv2.CAP_PROP_FPS))


        while True:
            
            ret_val, image = cam.read()
            
            if type(image)!=type(None):
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                cv2.putText(image,
                            "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.imshow('tf-pose-estimation result', image)
                fps_time = time.time()

                if cv2.waitKey(1) == 27:
                    break
                
                if humans:
                    #checking if two ears are visible in the image
                    s1 = set([16, 17])
                    if s1.issubset(humans[0].body_parts.keys()):
                        two_ears.append(1.0)
                    else:
                        two_ears.append(0.0)

                    #getting the coordinates of the eyes and the nose for triangulation
                    s2 = set([0,14,15])
                    if s2.issubset(humans[0].body_parts.keys()):
                        nose_coord.append([humans[0].body_parts[0].x, humans[0].body_parts[0].y])
                        left_eye_coord.append([humans[0].body_parts[14].x, humans[0].body_parts[14].y])
                        right_eye_coord.append([humans[0].body_parts[15].x, humans[0].body_parts[15].y])
                    

                        eye_left_x = humans[0].body_parts[14].x 
                        eye_right_x = humans[0].body_parts[15].x
                        nose_x = humans[0].body_parts[0].x
                        EyeDistance = eye_right_x - eye_left_x
                    
                        facing.append(float ( ( (nose_x > (eye_left_x + EyeDistance/4*3)) or  (nose_x < (eye_left_x + EyeDistance/4)) ) == False ) )
                    else:
                        facing.append(0.5)
                    
                else:
                    facing.append(0.0)
                    two_ears.append(0.0)
            
            else:
                cv2.destroyAllWindows()
                break

    return frames_per_second

detect(fps_time)
print (frames_per_second)

half_second = int(frames_per_second[0]/2)
print (half_second)

frame_result = np.add(two_ears, facing)/2

result_s = [sum(frame_result[i:i+half_second])/half_second for i in range(0, len(frame_result), half_second)]

gaze = [1 if x>0.5 else 0 for x in result_s]
clock = np.arange(0, len(gaze)*0.5, 0.5)

result_gaze = pd.DataFrame ({'clock':clock, 'gaze':gaze})
pd.DataFrame(result_gaze).to_csv('classifications/gaze-sample.csv', index=False)