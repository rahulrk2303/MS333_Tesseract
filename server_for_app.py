
import firebase_admin
from firebase_admin import credentials, db, storage, firestore
import cv2
import urllib
import numpy as np
from uuid import uuid4

from main_combined import predict_from_app


# cred = credentials.Certificate("vehicledetection-sih-e02ef-firebase-adminsdk.json")

# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://vehicledetection-sih-e02ef.firebaseio.com/',
# })

ref = db.reference('/Uploads')



db = firestore.client()

bucket = storage.bucket()


while True:
	uploads = ref.get()
	images = uploads["mohammedthowfiq2@gmail,com"]
	if 'ToBepProcessed' in uploads.keys():

		toprocess = uploads['ToBepProcessed']['imageName']

		# for i in images.keys():
		# print(images[i]['imageUrl'])
		resp = urllib.request.urlopen(images[toprocess]['imageUrl'])
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		print(toprocess)
		path = "uploads/"+toprocess + ".png"
		cv2.imwrite(path, image)
		# cv2.imshow("im", image)
		# cv2.waitKey(0)
		main_dict, out_path = predict_from_app(path)


		ref.child('mohammedthowfiq2@gmail,com').child(toprocess).child('outjson').set(main_dict) 



		imgpath = "outputs/"+toprocess + ".png"

		blob = bucket.blob(imgpath)
		# blob.make_public()

		new_token = uuid4()

		metadata  = {"firebaseStorageDownloadTokens": new_token}

		blob.metadata = metadata

		blob.upload_from_filename(filename=out_path, content_type='image/png')
		outurl = blob.public_url

		ref.child('mohammedthowfiq2@gmail,com').child(toprocess).child('outurl').set(outurl) 

		# ref.child('ToBepProcessed').child('imageName').set('')
		ref.child('ToBepProcessed').delete()

