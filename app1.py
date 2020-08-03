import os
import glob
from classify import prediction
import tensorflow as tf
import  _thread
import time
import json

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug import secure_filename

from main_combined import predict_from_web, predict_video_from_web, input_image_details, search_for_vehicle, add_suspect_to_db

app = Flask(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1




app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png', 'mp4', 'avi', 'MOV'])

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
	return render_template('index1.html')


@app.route('/upload', methods=['POST'])
def upload():
	file = request.files['file']
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)

		file_name_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(file_name_full_path)

	if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
		return redirect('claimimg')
	else:
		return redirect('claimvid')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)


# @app.route('/claim', methods=['POST'])
@app.route('/claimimg')
def predict_img():
	image_path = max(glob.glob(r'uploads/*'), key=os.path.getctime)
	out, out_path = predict_from_web(image_path)

	# for i in len(out):
	#     x = ""
	#     x += out[]
	# print(out)

	return render_template('front1.html', text = out, filename= out_path)


# @app.route('/claim', methods=['POST'])
@app.route('/claimvid')
def predict_vid():
	video_path = max(glob.glob(r'uploads/*'), key=os.path.getctime)
	out, json_path, out_path = predict_video_from_web(video_path)

	# out_path = max(glob.glob(r'uploads/out*.mp4'), key=os.path.getctime)
	# print(out_path)
	# time.sleep(5)

	# for i in len(out):
	#     x = ""
	#     x += out[]
	# print(out)

	return render_template('front_vid1.html', text = out , json= json_path, filename= out_path)



@app.route('/searchplate')
def searchplate():
	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
	f = open(json_path) 
	data = json.load(f)
	[search_out, map_out] = input_image_details(data, 0)
	search_out = "<h1>Search by plate number results </h1>" + search_out

	return render_template('search_result1.html', text = search_out , map = map_out , filename= json_path)
	# render_template("<h1>Hi</h1>")


@app.route('/searchvehicle')
def searchvehicle():
	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
	f = open(json_path) 
	data = json.load(f)
	[search_out, map_out]  = input_image_details(data, 1)
	search_out = "<h1>Search by vehicle make model color results </h1>" + search_out

	return render_template('search_result1.html', text = search_out , map = map_out , filename= json_path)
	# render_template("<h1>Hi</h1>")

@app.route('/addsuspect', methods=['POST', 'GET'])
def addsuspect():
	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
	f = open(json_path) 
	data = json.load(f)
	search_out = add_suspect_to_db(data)

	# search_out = "<h1>Search by vehicle make model color results </h1>" + search_out

	# return render_template('search_result.html', text = search_out , filename= json_path)
	return render_template('index1.html')
	# render_template("<h1>Hi</h1>")



def cleanDirectory(threadName,delay):
   # while True:
	   # time.sleep(delay)
	print ("Cleaning Up Directory")
	filelist = [ f for f in (os.listdir(app.config['UPLOAD_FOLDER']))  ]
	for f in filelist:
	 #os.remove("Uploads/"+f)
	 os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

	filelist = [ f for f in (os.listdir('outputs'))  ]
	for f in filelist:
	 #os.remove("Uploads/"+f)
	 os.remove(os.path.join('outputs', f))



if __name__ == '__main__':
	try:
		print("start")
		# _thread.start_new_thread( cleanDirectory, ("Cleaning Thread", 300, ) )
	except:
	   print("Error: unable to start thread" )
	app.run()
