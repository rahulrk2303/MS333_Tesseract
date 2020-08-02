from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from imutils.video import FPS
from imutils.video import FileVideoStream

from datetime import datetime, date

from threading import Thread
import concurrent.futures
from openalpr import Alpr
from car_make_model_color import Classifier
import json
from json2html import *

from damage import pred



# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db


from skimage.measure import compare_ssim as ssim
from color_classification import process_lp_color
from PIL import Image

import _thread


# Firebase end

# Setup ALPR

# alpr = Alpr("in", "openalpr.in_slow.conf", "runtime_data")
alpr = Alpr("in", "openalpr.in_slow.conf", "runtime_data")
if not alpr.is_loaded():
	print("Error loading OpenALPR")
	sys.exit(1)
alpr.set_top_n(5)
alpr.set_default_region("in")


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

def cvDrawBoxes_image(detections, img, counter):
	# Colored labels dictionary
	color_dict = {
		'person' : [0, 255, 255], 
		# 'motorbike' : [255, 255, 0], 
		'truck' : [0, 255, 0], 'bus' : [255, 0, 0],
		'car' : [0, 0, 255]
	}

	global classifier

	detections_list = []

	# print(len(detections))

	for detection in detections:
		if detection[1] > 0.3 :
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
					plate_conf_list = []



					if name_tag == 'person':
						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						cv2.putText(img, name_tag,
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)


					if boxed.shape[0] > 80 and boxed.shape[1] > 80 and name_tag in ['car', 'truck', 'bus']:  
						
						lp_color_type = ""

						
						colors, colors_conf = classifier.predict_color(boxed)

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

								plate_list.append(candidate['plate'])
								plate_conf_list.append(candidate['confidence'])

							
															

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


							top_left_x = min([p1[0],p2[0],p3[0],p4[0]])
							top_left_y = min([p1[1],p2[1],p3[1],p4[1]])
							bot_right_x = max([p1[0],p2[0],p3[0],p4[0]])
							bot_right_y = max([p1[1],p2[1],p3[1],p4[1]])
							plate_img = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

							plate_img2 = cv2.resize(plate_img, (50, 50), interpolation=cv2.INTER_LINEAR)
							plate_img2 = cv2.cvtColor(plate_img2, cv2.COLOR_BGR2RGB)

							plate_img2 = Image.fromarray(plate_img2)
							lp_color_type = process_lp_color(plate_img2)

							# print(lp_color)

							# cv2.imwrite("plate.png", plate_img)



							# cv2.imshow("plate", plate_img)
							# cv2.waitKey(0)

							pts = np.array(pts)
							pts = pts.reshape((-1,1,2))
							cv2.polylines(img,[pts],True,(0,255,0),thickness = 2)
							cv2.putText(img, str(lpr_candidate) + ' : ' + lp_color_type,
									(p1[0]+10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
									[0, 255, 0], 2)

							# plates_dict = {
							# 	# "plate_box": [p1,p2,p3,p4],
							# 	"plate": plate_list
							# }

							# print(plates_dict)



						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						cv2.putText(img,
									detection[0].decode() + " " + 
									colors[0],
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)


						detections_list_dict = {}

						if detection[0]:
							detections_list_dict["vehicle_type"] = name_tag
						
							detections_list_dict["colors"] = colors

							detections_list_dict['plate'] = plate_list
							detections_list_dict['plate_conf'] = plate_conf_list

							detections_list_dict['plate_type'] = lp_color_type

						detections_list.append(detections_list_dict)


	image_name = filename.split('/')[-1].split('.')[0]



	if detections_list:
		main_dict["image_name"] = image_name
		main_dict["detections"] = detections_list



	return img



def cvDrawBoxes_video(detections, img, counter):

	color_dict = {
		'bus' : [255, 0, 0],
		'car' : [0, 0, 255]
	}

	global classifier

	detections_list = []

	
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
					plate_conf_list = []
	

					# if boxed.shape[0] > 100 and boxed.shape[1] > 100 and name_tag in ['car', 'truck', 'bus', 'motorbike', 'person']:  
					if boxed.shape[0] > 100 and boxed.shape[1] > 100 and name_tag in color_dict.keys():  

						colors, colors_conf = classifier.predict_color(boxed)

						results = alpr.recognize_ndarray(boxed)
						lpr_candidate = ""
						lp_color_type = ""

						i = 0



						for plate in results['results']:
							i += 1
							# print("Plate #%d" % i)
							print("   %12s %12s" % ("Plate", "Confidence"))
							lpr_candidate = plate['candidates'][0]['plate']
							

							for candidate in plate['candidates']:

								

								prefix = "-"
								if candidate['matches_template']:
									prefix = "*"

								print(" %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
								# plate_list.append({
								# 	"number": candidate['plate'],
								# 	"plate_confidence": candidate['confidence'],
								# 	# "match_template": prefix
								# 	})

								plate_list.append(candidate['plate'])
								plate_conf_list.append(candidate['confidence'])





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

							top_left_x = min([p1[0],p2[0],p3[0],p4[0]])
							top_left_y = min([p1[1],p2[1],p3[1],p4[1]])
							bot_right_x = max([p1[0],p2[0],p3[0],p4[0]])
							bot_right_y = max([p1[1],p2[1],p3[1],p4[1]])
							plate_img = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

							plate_img2 = cv2.resize(plate_img, (50, 50), interpolation=cv2.INTER_LINEAR)
							plate_img2 = cv2.cvtColor(plate_img2, cv2.COLOR_BGR2RGB)

							plate_img2 = Image.fromarray(plate_img2)
							lp_color_type = process_lp_color(plate_img2)


							pts = np.array(pts)
							pts = pts.reshape((-1,1,2))
							cv2.polylines(img,[pts],True,(0,255,0),thickness = 2)
							cv2.putText(img, str(lpr_candidate) + ' : ' + lp_color_type,
									(p1[0]+10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)

							# plates_dict = {
							# 	# "plate_box": [p1,p2,p3,p4],
							# 	"plate": plate_list
							# }

							# print(plates_dict)

						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						cv2.putText(img,
									name_tag + ' ' +
									colors[0],
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)


						detections_list_dict = {}

						if detection[0]:
							detections_list_dict["vehicle_type"] = name_tag

							detections_list_dict["colors"] = colors

							detections_list_dict['plate'] = plate_list
							detections_list_dict['plate_conf'] = plate_conf_list

							detections_list_dict['plate_type'] = lp_color_type

						detections_list.append(detections_list_dict)


	video_name = filename.split('/')[-1].split('.')[0]


	if detections_list:
		main_dict['frames'].append({
			"frame": counter,
			"detections": detections_list,
			"datetime" : datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f"),
			"frametime": 1/ (main_dict['fps']/counter)
		})


		


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

		detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.3)    # Detection occurs at this line and return detections, for customize we can change the threshold.             

		# print("Frame no. ", format(counter))
		frame_time = 0

		image = cvDrawBoxes_image(detections, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), frame_time)               # Call the function cvDrawBoxes() for colored bounding box per class

		out_image_name = "out" + filename.split('/')[-1]

		out_path = "uploads/" + out_image_name

		# print(out_path)

		cv2.imwrite(out_path, image)

	cv2.destroyAllWindows()
	return out_path



def YOLO_video(filename):
   
	global metaMain, netMain, altNames
	
	#cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
	# cap = cv2.VideoCapture("../videos/camera11.mp4")
	print("[INFO] starting video file thread...")
	# fvs = FileVideoStream("../../../Downloads/deepdrive.mp4").start()

	fvs = FileVideoStream(filename).start()
	# cap = cv2.VideoCapture("test2.mp4")                             # Local Stored video detection - Set input video
	frame = fvs.read()
	# frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	# frames = []

	video = cv2.VideoCapture(filename)
	fps_video = video.get(cv2.CAP_PROP_FPS)
	skipframes = 2

	main_dict['fps'] = fps_video
	main_dict['skipframes'] = skipframes
	# print("FPS ACTUAL : ", format(fps_video))

	frame_height, frame_width = frame.shape[:2]
	darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network

	video_name = filename.split('/')[-1]

	out_path = 'uploads/out' + video_name

	out_writer = cv2.VideoWriter(out_path,
	cv2.VideoWriter_fourcc(*'avc1'), fps_video//skipframes, (frame_width,frame_height))


	counter = 1
	print("Starting the YOLO loop...")
	fps = FPS().start()

	prev_frame = frame

	# time.sleep(5)

	while fvs.more():

		frame = fvs.read()
		# frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

		if type(frame) is not np.ndarray:
			break
		else:
			if counter%skipframes == 0:

				simlarityIndex = ssim(cv2.cvtColor(
					cv2.resize(prev_frame, (200, 200), interpolation=cv2.INTER_LINEAR), 
					cv2.COLOR_BGR2GRAY), 
					cv2.cvtColor(cv2.resize(frame, (200, 200), interpolation=cv2.INTER_LINEAR), 
					cv2.COLOR_BGR2GRAY))

				if simlarityIndex > 0.9:
					print("Skipping similar frames")
					continue
				else:
					prev_frame = frame




				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
				frame_resized = cv2.resize(frame_rgb,
										   (frame_width, frame_height),
										   interpolation=cv2.INTER_LINEAR)

				darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

				detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.4)    # Detection occurs at this line and return detections, for customize we can change the threshold.             


				image = cvDrawBoxes_video(detections, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), counter)               # Call the function cvDrawBoxes() for colored bounding box per class
				# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# print((time.time()-prev_time))

				cv2.imshow('Demo', image)                                    # Display Image window
				cv2.waitKey(1)
				out_writer.write(image)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
				# cv2.imwrite('cars/frame' + str(counter//skipframes) + '.jpg', image)
				
			fps.update()
		counter+=1  

	fps.stop()

	fvs.stop()

	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	cv2.destroyAllWindows()
	# alpr.unload()
	return out_path


def save_json_image(image_name, main_dict):

	with open("results/" + image_name + ".json", 'w') as out:
		json.dump(main_dict, out, indent=4, separators=(',', ': ')) 

	# ref = db.reference('/query_images')
	# ref.child(image_name).set(main_dict)

	print("JSON stored to: results/" + image_name + ".json" ) 

	return



def save_json_video(video_name, main_dict):
	
	json_path = "results/" + video_name + ".json"

	with open(json_path, 'w') as out:
		json.dump(main_dict, out, indent=4, separators=(',', ': '))  

	print("JSON stored to: " + json_path ) 

	print("Uploaded to firebase")




def predict_from_web(fname):
	global filename, main_dict, alpr
	filename = fname
	main_dict = {}
	# generate_suspect_list()
	# _thread.start_new_thread(generate_suspect_list,())


	# alpr = Alpr("in", "openalpr.in_slow.conf", "runtime_data")

	# alpr.unload()
	alpr = Alpr("in", "openalpr.in_slow.conf", "runtime_data")
	if not alpr.is_loaded():
		print("Error loading OpenALPR")
		sys.exit(1)
	alpr.set_top_n(5)
	alpr.set_default_region("in")

	out_path = YOLO_image(filename)                                                           # Calls the main function YOLO()
	
	image_name = filename.split('/')[-1].split('.')[0]

	print(image_name)

	main_dict['location'] = filename  
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S%f")

	# main_dict['query'] = image_name

	_thread.start_new_thread(save_json_image, (image_name, main_dict))
	# save_json_image (image_name, main_dict)

	# return json.dumps(main_dict,indent=2)
	return json2html.convert(json = main_dict), out_path

# alpr.unload()
	


def predict_video_from_web(fname):  


	# _thread.start_new_thread(generate_suspect_list,())


	global filename, alpr, main_dict
	filename = fname

	main_dict = {}

	
	video_name = filename.split('/')[-1].split('.')[0]

	# alpr.unload()
	alpr = Alpr("in", "openalpr.in.conf", "runtime_data")
	if not alpr.is_loaded():
		print("Error loading OpenALPR")
		sys.exit(1)
	alpr.set_top_n(5)
	alpr.set_default_region("in")


	main_dict['location'] = filename
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
	main_dict['camera_id'] = video_name
	main_dict['frames'] = []

	# main_dict[filename.split('/')[-1].split('.')[0]] = {}

	out_path = YOLO_video(filename)  

	json_path = "results/" + video_name + ".json"


	_thread.start_new_thread(save_json_video, (video_name, main_dict))


	return json2html.convert(json = main_dict), json_path, out_path

	# alpr.unload()   




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


# if __name__ == "__main__":  	

# 	while True:
# 		print("Enter path to image : ")
# 		filename = "./images/" + input()

# 		YOLO_image(filename)                                                           # Calls the main function YOLO()

# 		image_name = filename.split('/')[-1].split('.')[0]

# 		main_dict['location'] = filename  
# 		main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
# 		main_dict['query'] = image_name


# 		with open("results/" + image_name + ".json", 'w') as out:
# 			json.dump(main_dict, out, indent=4, separators=(',', ': ')) 

# 		# ref = db.reference('/query_images')
# 		# ref.child(image_name).set(main_dict)

# 		print("JSON stored to: results/" + image_name + ".json" ) 

# 		ref = db.reference('/query_images')
# 		ref.child(image_name).set(main_dict)


# 		print("Uploaded to firebase")

# 	alpr.unload()


if __name__ == "__main__":  

	# filename = "./videos/shantha2.mp4"
	filename = "./videos/guleba.mp4"

	video_name = filename.split('/')[-1].split('.')[0]


	alpr.unload()
	alpr = Alpr("in", "openalpr.in.conf", "runtime_data")
	if not alpr.is_loaded():
		print("Error loading OpenALPR")
		sys.exit(1)
	alpr.set_top_n(5)
	alpr.set_default_region("in")


	main_dict['location'] = filename
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
	main_dict['camera_id'] = video_name
	main_dict['frames'] = []

	# main_dict[filename.split('/')[-1].split('.')[0]] = {}

	YOLO_video(filename)  

	with open("results/" + video_name + ".json", 'w') as out:
		json.dump(main_dict, out, indent=4, separators=(',', ': '))  

	print("JSON stored to: results/" + video_name + ".json" ) 

	print("Uploaded to firebase")

	alpr.unload()    

