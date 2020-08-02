import os
import glob
from classify import prediction
import tensorflow as tf
import  _thread
import time
import panads as pd
import json
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug import secure_filename

from main_combined import predict_from_web, predict_video_from_web



# Initialize the Flask application
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
# No caching at all for API endpoints.
# @app.after_request
# def add_header(response):
#     # response.cache_control.no_store = True
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response



# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png', 'mp4', 'avi', 'MOV'])
# app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
	return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
	# Get the name of the uploaded file
	file = request.files['file']
	# Check if the file is one of the allowed types/extensions
	if file and allowed_file(file.filename):
		# Make the filename safe, remove unsupported chars
		filename = secure_filename(file.filename)
		# filename  = "img"+str(len(os.listdir(app.config['UPLOAD_FOLDER']))+1)+'.png'
		# Move the file form the temporal folder to
		# the upload folder we setup
		file_name_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(file_name_full_path)
		# Redirect the user to the uploaded_file route, which
		# will basicaly show on the browser the uploaded file
	# return render_template('upload_success.html')
	if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
		return redirect('claimimg')
	else:
		return redirect('claimvid')

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
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

	return render_template('front.html', text = out, filename= out_path)


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

	return render_template('front_vid.html', text = out , json= json_path, filename= out_path)



# @app.route('/searchplate')
# def searchplate():
# 	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
# 	f = open(json_path) 
# 	data = json.load(f)
# 	search_out = input_image_details(data, 0)
# 	search_out = "<h1>Search by plate number results </h1>" + search_out

# 	return render_template('search_result.html', text = search_out , filename= json_path)
# 	# render_template("<h1>Hi</h1>")


# @app.route('/searchvehicle')
# def searchvehicle():
# 	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
# 	f = open(json_path) 
# 	data = json.load(f)
# 	search_out = input_image_details(data, 1)
# 	search_out = "<h1>Search by vehicle make model color results </h1>" + search_out

# 	return render_template('search_result.html', text = search_out , filename= json_path)
# 	# render_template("<h1>Hi</h1>")

# @app.route('/addsuspect', methods=['POST', 'GET'])
# def addsuspect():
# 	json_path = max(glob.glob(r'results/*'), key=os.path.getctime)
# 	f = open(json_path) 
# 	data = json.load(f)
# 	search_out = add_suspect_to_db(data)

# 	# search_out = "<h1>Search by vehicle make model color results </h1>" + search_out

# 	# return render_template('search_result.html', text = search_out , filename= json_path)
# 	return render_template('index.html')
# 	# render_template("<h1>Hi</h1>")



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
