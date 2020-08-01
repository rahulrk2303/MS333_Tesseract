import cv2
import numpy as np
import time
import darknet
import os


def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax

dd = 0
# classifier = Classifier()
# prev_time_lp = 0

def cvDrawBoxes(detections, img):
	# Colored labels dictionary
	color_dict = {
		'person' : [0, 255, 255], 'car' : [0, 0, 255], 'motorbike' : [255, 255, 0], 
		'bus' : [255, 0, 0], 'truck' : [0, 255, 0]
	}

	# global dd, classifier
	
	for detection in detections:
		if detection[1] > 0.4 :
			x, y, w, h = detection[2][0],\
				detection[2][1],\
				detection[2][2],\
				detection[2][3]
			name_tag = str(detection[0].decode())
			for name_key, color_val in color_dict.items():
				if name_key == name_tag:
					color = color_val 
					xmin, ymin, xmax, ymax = convertBack(
					float(x), float(y), float(w), float(h))
					
				
					pt1 = (xmin, ymin)
					pt2 = (xmax, ymax)
					cv2.rectangle(img, pt1, pt2, color, 2)

	return img


configPath = "./cfg/yolov4.cfg"                               
weightPath = "./weights/yolov4.weights"                               
metaPath = "./cfg/coco.data"   
netMain = ''                                
if not os.path.exists(configPath):                             
	raise ValueError("Invalid config path `" +
					 os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
	raise ValueError("Invalid weight path `" +
					 os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
	raise ValueError("Invalid data file path `" +
					 os.path.abspath(metaPath)+"`")
                                            
netMain = darknet.load_net_custom(configPath.encode( 
		"ascii"), weightPath.encode("ascii"), 0, 1)            
metaMain = darknet.load_meta(metaPath.encode("ascii"))



frame = cv2.imread("test.png")
frame_height, frame_width = frame.shape[:2]
darknet_image = darknet.make_image(frame_width, frame_height, 3)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb,(frame_width, frame_height),
										   interpolation=cv2.INTER_LINEAR)

darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                

detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)                    
image = cvDrawBoxes(detections, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)) 

cv2.imshow("test", image)
cv2.waitKey(0)