from ctypes import *                                       # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from imutils.video import FPS
from imutils.video import FileVideoStream

from datetime import datetime

from threading import Thread
import concurrent.futures
from openalpr import Alpr
from car_make_model_color import Classifier
import json
from json2html import *

from damage import pred

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Firebase start

cred = credentials.Certificate("sihdb-895b3-firebase-admin-it.json")

firebase_admin.initialize_app(cred, {
    'databaseURL':'https://sihdb-895b3.firebaseio.com/'
})

# Firebase end

# Setup ALPR

alpr = Alpr("in", "openalpr.in_slow.conf", "runtime_data")
if not alpr.is_loaded():
	print("Error loading OpenALPR")
	sys.exit(1)
alpr.set_top_n(10)
alpr.set_default_region("in")

dd = 0
classifier = Classifier()

netMain = None
metaMain = None
altNames = None

filename = ""


def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax

main_dict = {}

def cvDrawBoxes(detections, img, counter):
	# Colored labels dictionary
	color_dict = {
		'person' : [0, 255, 255], 'motorbike' : [255, 255, 0], 
		'truck' : [0, 255, 0], 'bus' : [255, 0, 0],
		'car' : [0, 0, 255]
	}

	global dd, classifier

	detections_list = []

	# print(len(detections))

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
					
					boxed = img[ymin:ymax, xmin:xmax]
					
					plate_list = []
					plates_dict = {}


					if boxed.shape[0] > 50 and boxed.shape[1] > 50 and name_tag in ['car', 'truck', 'bus', 'motorbike', 'person']:  

						damage = pred(boxed) # Damage detection
						

						color_result = classifier.predict_color(boxed)
						mm_result = classifier.predict_mm(boxed)

						results = alpr.recognize_ndarray(boxed)
						lpr_candidate = ""

						i = 0

						# print("TIME : ", format(frame_time))

						# print("\n\n\n\n\n")
						# print(results)	
						# print("\n\n\n\n\n")
						

						for plate in results['results']:
							i += 1
							# print("Plate #%d" % i)
							print("   %12s %12s" % ("Plate", "Confidence"))
							lpr_candidate = plate['candidates'][0]['plate']
							
							# print(plate['candidates'])

							

							for candidate in plate['candidates']:
								prefix = "-"
								if candidate['matches_template']:
									prefix = "*"

								print(" %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
								plate_list.append({
									"number": candidate['plate'],
									"plate_confidence": candidate['confidence'],
									"match_template": prefix
									})

							# print(plate_list)

							coordinates = results['results'][0]['coordinates']
							p1, p2, p3, p4 = list(coordinates[0].values()), list(coordinates[1].values()), list(coordinates[2].values()), list(coordinates[3].values())
							p1[0] += xmin
							p2[0] += xmin
							p3[0] += xmin
							p4[0] += xmin
							p1[1] += ymin
							p2[1] += ymin
							p3[1] += ymin
							p4[1] += ymin
							pts = [p1,p2,p3,p4]
							pts = np.array(pts)
							pts = pts.reshape((-1,1,2))
							cv2.polylines(img,[pts],True,(0,255,0),thickness = 2)
							# cv2.putText(img, str(lpr_candidate),
							# 		(p1[0]+10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
							# 		color, 2)

							plates_dict = {
								"plate_box": [p1,p2,p3,p4],
								"plate": plate_list
							}

							# print(plates_dict)

						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						# cv2.putText(img,
						# 			detection[0].decode() + " " + 
						# 			mm_result[0]['make'] + " " + 
						# 			mm_result[0]['model'] + " " +
						# 			color_result[0]['color'],
						# 			(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
						# 			color, 2)

						# detections_list.append({
						# 	"vehicle_type": detection[0].decode(),
						# 	"bounding_box": [list(pt1), list(pt2)],
						# 	"confidence": detection[1],
						# 	"color": color_result,
						# 	"make_model": mm_result,
							
						# 	"plates": plates_dict
						# })

						detections_list_dict = {}

						if detection[0]:
							detections_list_dict["vehicle_type"] = name_tag
							detections_list_dict["bounding_box"] = [list(pt1), list(pt2)]
							detections_list_dict["confidence"] = detection[1]

						if name_tag == 'car' or name_tag == 'truck' or name_tag == 'bus':
							detections_list_dict["color"] = color_result
							detections_list_dict["make_model"] = mm_result
							detections_list_dict["damage"] = damage

						if plates_dict:
							detections_list_dict['plates'] = plates_dict

						detections_list.append(detections_list_dict)

						dd+=1
	# print(detections_list)

	# sec = str(frame_time).split('.')[0]
	# millisec = str(frame_time).split('.')[1]
	# ttt = sec + '_' + millisec
	image_name = filename.split('/')[-1].split('.')[0]


	if detections_list:
		main_dict["image"] = {
			"image_name": image_name,
			"detections": detections_list
		}


	# if detections_list:
	# 	main_dict[filename.split('/')[-1].split('.')[0]]["time"]= [
	# 			str(frame_time), detections_list
	# 		]

	# push_data = {
	# 	str(frame_time): {
	# 		"detections": detections_list
	# 	}
	# }
		


	return img


def YOLO_image(filename):
   
	global metaMain, netMain, altNames

	# filename = "./images/img1.png"


	frame = cv2.imread(filename)


	if type(frame) is not np.ndarray:
		print("Invalid image")
	else:

		frame_height, frame_width = frame.shape[:2]
		darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
		frame_resized = cv2.resize(frame_rgb,
								   (frame_width, frame_height),
								   interpolation=cv2.INTER_LINEAR)

		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

		detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.             

		# print("Frame no. ", format(counter))
		frame_time = 0

		image = cvDrawBoxes(detections, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), frame_time)               # Call the function cvDrawBoxes() for colored bounding box per class
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# print((time.time()-prev_time))

		# cv2.imshow('Demo', image)                                    # Display Image window
		
		# cv2.waitKey(0)

		# out_image_path = "uploads/"
		out_image_name = "out" + filename.split('/')[-1]

		out_path = "uploads/" + out_image_name

		print(out_path)

		cv2.imwrite(out_path, image)

	cv2.destroyAllWindows()
	return out_path


configPath = "./cfg/yolov4-tiny.cfg"                                 # Path to cfg
weightPath = "./weights/yolov4-tiny.weights"                                 # Path to weights
metaPath = "./cfg/coco.data"                                    # Path to meta data
if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
	raise ValueError("Invalid config path `" +
					 os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
	raise ValueError("Invalid weight path `" +
					 os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
	raise ValueError("Invalid data file path `" +
					 os.path.abspath(metaPath)+"`")
if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
	netMain = darknet.load_net_custom(configPath.encode( 
		"ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
if metaMain is None:
	metaMain = darknet.load_meta(metaPath.encode("ascii"))


if __name__ == "__main__":  	

	while True:
		print("Enter path to image : ")
		filename = "./images/" + input()

		YOLO_image(filename)                                                           # Calls the main function YOLO()

		image_name = filename.split('/')[-1].split('.')[0]

		main_dict['location'] = filename  
		main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		main_dict['query'] = image_name


		# with open("results/" + image_name + ".json", 'w') as out:
		# 	json.dump(main_dict, out, indent=4, separators=(',', ': ')) 

		# print("JSON stored to: results/" + image_name + ".json" ) 

		# ref = db.reference('/query_images')
		# ref.child(image_name).set(main_dict)


		# print("Uploaded to firebase")

	alpr.unload()


def predict_from_web(fname):
	global filename, main_dict
	filename = fname
	main_dict = {}

	out_path = YOLO_image(filename)                                                           # Calls the main function YOLO()
	
	image_name = filename.split('/')[-1].split('.')[0]

	main_dict['location'] = filename  
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S%f")
	main_dict['query'] = image_name


	with open("results/" + image_name + ".json", 'w') as out:
		json.dump(main_dict, out, indent=4, separators=(',', ': ')) 

		# ref = db.reference('/query_images')
		# ref.child(image_name).set(main_dict)

	print("JSON stored to: results/" + image_name + ".json" ) 

	ref = db.reference('/query_images')
	ref.child(image_name).set(main_dict)


	print("Uploaded to firebase")

	# return json.dumps(main_dict,indent=2)
	return json2html.convert(json = main_dict), out_path

# alpr.unload()
	
