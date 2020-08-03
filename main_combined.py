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

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


from skimage.measure import compare_ssim as ssim
from color_classification import process_lp_color
from PIL import Image

import _thread

# Firebase start

# Vykunth account
# cred = credentials.Certificate("sihdb-895b3-firebase-admin-it.json")

# firebase_admin.initialize_app(cred, {
#     'databaseURL':'https://sihdb-895b3.firebaseio.com/'
# })

# Tesseract account
cred = credentials.Certificate("vehicledetection-sih-e02ef-firebase-adminsdk.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://vehicledetection-sih-e02ef.firebaseio.com/',
    'storageBucket': 'vehicledetection-sih-e02ef.appspot.com'

})

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
					suspect_car = False
					expired = []
					exp_out = ''


					# plates_dict = {}


					if name_tag == 'person':
						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						cv2.putText(img, name_tag,
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)


					if boxed.shape[0] > 80 and boxed.shape[1] > 80 and name_tag in ['car', 'truck', 'bus']:  

						damage = pred(boxed) # Damage detection
						
						lp_color_type = ""

						
						colors, colors_conf = classifier.predict_color(boxed)
						make_model, make_model_conf = classifier.predict_mm(boxed)

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

								if candidate['plate'] in suspect_list:
									# print("SUSPECT")
									suspect_car = True



								plex = is_expired(candidate['plate'])
								expired.append(plex)
								if plex == 'Valid' or plex == 'Expired':
									lpr_candidate = candidate['plate']


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

							# print(plate_list)

							if 'Valid' in expired:
								exp_out = 'Valid'
							elif 'Expired' in expired:
								exp_out = 'Expired'
							else:
								exp_out = 'Unregistered'

							if suspect_car:
								cv2.putText(img, "Suspect found",
									(40,50), cv2.FONT_HERSHEY_SIMPLEX, 2,
									color, 3)
								print("SUSPECT FOUND")

							
															

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
									[0,0,255], 2)

							# plates_dict = {
							# 	# "plate_box": [p1,p2,p3,p4],
							# 	"plate": plate_list
							# }

							# print(plates_dict)



						pt1 = (xmin, ymin)
						pt2 = (xmax, ymax)
						cv2.rectangle(img, pt1, pt2, color, 2)
						cv2.putText(img,
									# detection[0].decode() + " " + 
									make_model[0] + " " +
									colors[0],
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)

						if lpr_candidate:
							cv2.putText(img, "Insurance " + exp_out,
										(pt1[0]+10, pt2[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
										color, 2)

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
							# detections_list_dict["bounding_box"] = [list(pt1), list(pt2)]
							# detections_list_dict["confidence"] = detection[1]

						# if name_tag == 'car' or name_tag == 'truck' or name_tag == 'bus':
							detections_list_dict["make_model"] = make_model
							# detections_list_dict["make_model_conf"] = make_model_conf

							detections_list_dict["colors"] = colors
							# detections_list_dict["colors_conf"] = colors_conf
							
							detections_list_dict["damage"] = damage

						# if plates_dict:
							detections_list_dict['plate'] = plate_list
							detections_list_dict['plate_conf'] = plate_conf_list

							detections_list_dict['plate_type'] = lp_color_type

						# if suspect_car == True:
							detections_list_dict['suspect'] = suspect_car
							detections_list_dict['insurance'] = exp_out

						detections_list.append(detections_list_dict)

						
	# print(detections_list)

	# sec = str(frame_time).split('.')[0]
	# millisec = str(frame_time).split('.')[1]
	# ttt = sec + '_' + millisec
	image_name = filename.split('/')[-1].split('.')[0]


	# if detections_list:
	# 	main_dict["image"] = {
	# 		"image_name": image_name,
	# 		"detections": detections_list
	# 	}

	if detections_list:
		main_dict["image_name"] = image_name
		main_dict["detections"] = detections_list


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



def cvDrawBoxes_video(detections, img, counter):
	# Colored labels dictionary
	color_dict = {
		# 'person' : [0, 255, 255], 'motorbike' : [255, 255, 0], 
		# 'truck' : [0, 255, 0], 
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
					suspect_car = False
					expired = []
					exp_out = ''

					# plates_dict = {}


					# if boxed.shape[0] > 100 and boxed.shape[1] > 100 and name_tag in ['car', 'truck', 'bus', 'motorbike', 'person']:  
					if boxed.shape[0] > 100 and boxed.shape[1] > 100 and name_tag in color_dict.keys():  

						# damage = pred(boxed) # Damage detection
						

						colors, colors_conf = classifier.predict_color(boxed)
						make_model, make_model_conf = classifier.predict_mm(boxed)

						results = alpr.recognize_ndarray(boxed)
						lpr_candidate = ""
						lp_color_type = ""

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

								if candidate['plate'] in suspect_list:
									# print("SUSPECT")
									suspect_car = True
									cv2.putText(img, "Suspect found",
									(40,50), cv2.FONT_HERSHEY_SIMPLEX, 2,
									[0, 0, 255], 3)

								plex = is_expired(candidate['plate'])
								expired.append(plex)
								if plex == 'Valid' or plex == 'Expired':
									lpr_candidate = candidate['plate']

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

							# print(plate_list)

							if 'Valid' in expired:
								exp_out = 'Valid'
							elif 'Expired' in expired:
								exp_out = 'Expired'
							else:
								exp_out = 'Unregistered'



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
									# mm_result[0]['make'] + " " + 
									# mm_result[0]['model'] + " " +
									# color_result[0]['color'],
									make_model[0] + " " +
									colors[0],
									(pt1[0]+10, pt1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
									color, 2)

						if lpr_candidate:
							cv2.putText(img, "Insurance " + exp_out,
										(pt1[0]+10, pt2[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
										color, 2)

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
							# detections_list_dict["bounding_box"] = [list(pt1), list(pt2)]
							# detections_list_dict["confidence"] = detection[1]

						# if name_tag == 'car' or name_tag == 'truck' or name_tag == 'bus':
							detections_list_dict["make_model"] = make_model
							# detections_list_dict["make_model_conf"] = make_model_conf

							detections_list_dict["colors"] = colors
							# detections_list_dict["colors_conf"] = colors_conf
							

						# if plates_dict:
							detections_list_dict['plate'] = plate_list
							detections_list_dict['plate_conf'] = plate_conf_list

							detections_list_dict['plate_type'] = lp_color_type

							detections_list_dict['suspect'] = suspect_car
							detections_list_dict['insurance'] = exp_out



						detections_list.append(detections_list_dict)

						
	# print(detections_list)

	# sec = str(frame_time).split('.')[0]
	# millisec = str(frame_time).split('.')[1]
	# ttt = sec + '_' + millisec
	video_name = filename.split('/')[-1].split('.')[0]


	# if detections_list:
	# 	main_dict[video_name][counter] = {
	# 		"detections": detections_list
	# 	}

	if detections_list:
		main_dict['frames'].append({
			"frame": counter,
			"detections": detections_list,
			"datetime" : datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f"),
			"frametime": 1/ (main_dict['fps']/counter)
		})


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

		detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.3)    # Detection occurs at this line and return detections, for customize we can change the threshold.             

		# print("Frame no. ", format(counter))
		frame_time = 0

		image = cvDrawBoxes_image(detections, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), frame_time)               # Call the function cvDrawBoxes() for colored bounding box per class
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# print((time.time()-prev_time))

		# cv2.imshow('Demo', image)                                    # Display Image window
		
		# cv2.waitKey(0)

		# out_image_path = "uploads/"
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

				if simlarityIndex > 0.8:
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

				# print("Frame no. ", format(counter))
				# frame_time = 1/ (fps_video/counter)

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

	ref = db.reference('/query_images')
	ref.child(image_name).set(main_dict)

	print("Uploaded to firebase")
	return

cctv = db.reference('/cctv_cameras')
suspects = db.reference('/suspect_vehicles')


def save_json_video(video_name, main_dict):
	
	json_path = "results/" + video_name + ".json"

	with open(json_path, 'w') as out:
		json.dump(main_dict, out, indent=4, separators=(',', ': '))  

	print("JSON stored to: " + json_path ) 

	cctv.child(video_name).set(main_dict)

	print("Uploaded to firebase")




def predict_from_web(fname):
	global filename, main_dict, alpr
	filename = fname
	main_dict = {}
	# generate_suspect_list()
	_thread.start_new_thread(generate_suspect_list,())


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
	# loc = location_dict[image_name.split('_')[0]]


	main_dict['location'] = image_name 
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S%f")
	# main_dict['lat'] = loc[0]  
	# main_dict['lon'] = loc[1] 
	# main_dict['query'] = image_name

	_thread.start_new_thread(save_json_image, (image_name, main_dict))
	# save_json_image (image_name, main_dict)

	# return json.dumps(main_dict,indent=2)
	return json2html.convert(json = main_dict), out_path

# alpr.unload()
	


def predict_video_from_web(fname):  

	# filename = "./videos/shantha2.mp4"
	# filename = "./videos/shantha2.mp4"

	_thread.start_new_thread(generate_suspect_list,())


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

	loc = location_dict[video_name.split('_')[0]]
	main_dict['location'] = loc[2]
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
	main_dict['camera_id'] = video_name
	main_dict['frames'] = []

	main_dict['lat'] = loc[0]  
	main_dict['lon'] = loc[1] 

	# main_dict[filename.split('/')[-1].split('.')[0]] = {}

	out_path = YOLO_video(filename)  

	json_path = "results/" + video_name + ".json"


	_thread.start_new_thread(save_json_video, (video_name, main_dict))


	return json2html.convert(json = main_dict), json_path, out_path

	# alpr.unload()   


def predict_from_app(fname):
	global filename, main_dict, alpr
	filename = fname
	main_dict = {}

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

	# print(image_name)
	# loc = location_dict[image_name.split('_')[0]]

	main_dict['location'] = filename 
	main_dict['timestamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S%f")
	# main_dict['lat'] = loc[0]  
	# main_dict['lon'] = loc[1]  

	# main_dict['query'] = image_name

	# _thread.start_new_thread(save_json_image_app, (image_name, main_dict))
	# save_json_image (image_name, main_dict)

	# return json.dumps(main_dict,indent=2)
	# return json2html.convert(json = main_dict), out_path
	return main_dict, out_path
	

vahan_json = open('sample_vahan_db/vahan_db.json') 
vahan = json.load(vahan_json)
today = date.today()


location_dict = {
	'cctv1' : [28.608177, 77.216060,'Location 1'],
	'cctv2' : [28.610358, 77.212281,'Location 2'],
	'cctv3' : [28.609293, 77.209213,'Location 3'],
	'cctv4' : [28.610612, 77.205984,'Location 4'],
	'cctv5' : [28.611203, 77.199300,'Location 5'],
	'cctv6' : [28.605193, 77.198885,'Location 6'],
	'cctv7' : [28.602480, 77.193832,'Location 7'],
	'cctv8' : [28.605993, 77.187534,'Location 8'],

	'cctv9' : [28.584076, 77.151148,'Location 9'],
	'cctv10' : [28.582371, 77.154195,'Location 10'],
	'cctv11' : [28.583068, 77.149614,'Location 11'],
	'cctv12' : [28.579083, 77.148208,'Location 12'],
	'cctv13' : [28.579262, 77.146974,'Location 13'],
	
	'cctv14' : [28.572088, 77.161989,'Location 14'],
	'cctv15' : [28.578564, 77.175980,'Location 15'],
	'cctv16' : [28.570217, 77.184863,'Location 16'],
	'cctv17' : [28.565802, 77.181598,'Location 17'],
	'cctv18' : [28.562415, 77.188702,'Location 18']
}


def is_expired(lp):
	if lp in vahan.keys():
		dd = vahan[lp]['insurance_expiry']
		exp = dd.split('-')
		exp_date = date(int(exp[2]), int(exp[1]), int(exp[0]))
		# print(exp_date)
		# print(today)
		if exp_date < today:
			return 'Expired'
		else:
			return 'Valid'	
	else:
		return 'Unregistered'

import plotly.express as px
import plotly.graph_objects as go
import plotly
# import pandas as pd

def search_for_vehicle(search_plates, search_color, search_makemodel, plate_or_vehicle):
	plate_out = []
	car_out = []
	videos = cctv.get()

	lat_list = []
	lon_list = []
	loc_list = []


	for video in videos.keys():
		for i in range(len(videos[video]['frames'])):
			for j in range(len(videos[video]['frames'][i]['detections'])):
				car_match = False
				plate_match = False

				if plate_or_vehicle == 1:	
					for l in range(len(videos[video]['frames'][i]['detections'][j]['colors'])):
						if videos[video]['frames'][i]['detections'][j]['colors'][l] in search_color and \
						videos[video]['frames'][i]['detections'][j]['make_model'][l] in search_makemodel:
							car_match=True

						else:
							car_match= False

						if car_match:
							found = {}
							found['location'] = videos[video]['location']
							if 'plate' in videos[video]['frames'][i]['detections'][j].keys():
								found['plate'] = videos[video]['frames'][i]['detections'][j]['plate']
							else:
								found['plate'] = []
							found['make_model'] = videos[video]['frames'][i]['detections'][j]['make_model']
							found['colors'] = videos[video]['frames'][i]['detections'][j]['colors']
							found['frametime'] = videos[video]['frames'][i]['frametime']
							found['datetime'] = videos[video]['frames'][i]['datetime']
							found['lat'] = videos[video]['lat']
							found['lon'] = videos[video]['lon']
							lat_list.append(found['lat'])
							lon_list.append(found['lon'])
							loc_list.append(found['location'])
							car_out.append(found)
							# plate_match = True
							break
				

				elif 'plate' in videos[video]['frames'][i]['detections'][j].keys():
					for l in range(len(videos[video]['frames'][i]['detections'][j]['plate'])):
						if videos[video]['frames'][i]['detections'][j]['plate'][l] in search_plates:
							# print(videos[video]['frames'][i]['frame'])

							found = {}
							found['location'] = videos[video]['location']
							found['plate'] = videos[video]['frames'][i]['detections'][j]['plate']
							found['make_model'] = videos[video]['frames'][i]['detections'][j]['make_model']
							found['colors'] = videos[video]['frames'][i]['detections'][j]['colors']
							found['frametime'] = videos[video]['frames'][i]['frametime']
							found['datetime'] = videos[video]['frames'][i]['datetime']
							found['lat'] = videos[video]['lat']
							found['lon'] = videos[video]['lon']
							lat_list.append(found['lat'])
							lon_list.append(found['lon'])
							loc_list.append(found['location'])
							plate_out.append(found)

							plate_match = True
							break
						else:
							plate_match = False

				


					# if proceed == True:
					# 	for k in range(len(videos[video]['frames'][i]['detections'][j]['plate'])):
					# 		if videos[video]['frames'][i]['detections'][j]['plate'][k] in search_plates:
					# 			found = {}
					# 			found['plate'] = videos[video]['frames'][i]['detections'][j]['plate'][k]
					# 			found['make_model'] = videos[video]['frames'][i]['detections'][j]['make_model']
					# 			found['colors'] = videos[video]['frames'][i]['detections'][j]['colors']
					# 			found['location'] = videos[video]['location']
					# 			found['frametime'] = videos[video]['frames'][i]['frametime']
					# 			found['datetime'] = videos[video]['frames'][i]['datetime']
					# 			out.append(found)
					# 			# print(out)


	# for i in plate_out:
	# 	print(i)
	# 	print('\n')

	# print("len: ", len(plate_out))

	# return (json2html.convert(json = plate_out))                                                 
	# return plate_out, car_out

	map_out =''

	if lat_list:
		fig = px.scatter_mapbox(lat=lat_list, lon=lon_list, hover_name=loc_list,
			color_discrete_sequence=["red"]*len(lat_list), zoom=14, height=500, size=[20]*len(lat_list))

		fig.update_layout(mapbox_style="streets", mapbox_accesstoken="pk.eyJ1IjoicmFodWxyazIzMDMiLCJhIjoiY2tkZDhka2dyMDl1ODJ3bnIyMGIza2wxZCJ9.zlNfmSZyo5K3KcGEvclZ-A")
		fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
		# plotly.offline.plot(fig, filename='file.html')
		map_out = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')

	if plate_or_vehicle == 0:
		return plate_out, map_out
	else:
		return car_out, map_out                                              


def input_image_details(data, plate_or_vehicle):
	search_plates = []
	search_color = []
	search_makemodel = []
	search_out = []
	for i in range(len(data['detections'])):
		# print(queries[query]['detections'][i])
		print(data['image_name'])
		if 'plate' in data['detections'][i].keys():
			search_plates = data['detections'][i]['plate']
			search_color = data['detections'][i]['colors']
			search_makemodel = data['detections'][i]['make_model']
			outt = search_for_vehicle(search_plates, search_color, search_makemodel, plate_or_vehicle)
			search_out.append(outt[0])
	return (json2html.convert(json = search_out), outt[1])                                                 


def add_suspect_to_db(data):
	suspect_plates = []
	for i in range(len(data['detections'])):
		# print(queries[query]['detections'][i])
		# print(data['image_name'])
		if 'plate' in data['detections'][i].keys():
			for j in data['detections'][i]['plate']:
				print(j)
				suspects.child(j).set('suspect')
	

suspect_list = []

def generate_suspect_list():
	global suspect_list
	suspect_list = suspects.get().keys()
	# suspect_list = list(set(suspect_list))
	# print(suspect_list_db)




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
	filename = "./videos/dev.mp4"

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

	ref = db.reference('/cctv_cameras')
	ref.child(video_name).set(main_dict)


	print("Uploaded to firebase")

	alpr.unload()    

