
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
from event import NewImageEvent
from azure.eventhub import EventHubProducerClient, EventData
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
#from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType, FaceAttributeType

credential = DefaultAzureCredential()
storage_client = BlobServiceClient("https://storageandidentity.blob.core.windows.net", credential)
eventhub_client = EventHubProducerClient("pythonimagecapture.servicebus.windows.net", "imagecapture", credential)

# Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
# This key will serve all examples in this document.
KEY = os.environ['FACE_SUBSCRIPTION_KEY']

# Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = os.environ['FACE_ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
faces_detected = 0


def analyseImage(image_name, frame):
	# Detect a face in an image that contains a single face
	emotions = ""
	with open(image_name, 'rb') as image_file:
		detected_faces = face_client.face.detect_with_stream(image_file, return_face_attributes=FaceAttributeType.emotion)
		if detected_faces:
			faces_detected = faces_detected  + 1
			# Display the detected face ID in the first single-face image.
			# Face IDs are used for comparison to faces (their IDs) detected in other images.
			for face in detected_faces:
				emotions = face.face_attributes.emotion
			print("Faces detected: " + faces_detected)
	return emotions

def uploadToStorage(image_name, frame):
	container_client = storage_client.get_container_client("cameraimages")
	try:
		container_client.create_container()
	except:
		#Container already exists
		print("Container already exists")
	with open(image_name, 'rb') as image_file:
		container_client.upload_blob(image_name, image_file.read(), overwrite=True)
		#blob_client.upload_blob(image_file.read())

def sendEvent(image_data):
	batch = eventhub_client.create_batch(max_size_in_bytes=10000)
	
	data = EventData(image_data.encode())
	try:
		batch.add(data)
	except ValueError:
		print("Can't add data to batch")

	eventhub_client.send_batch(batch)
	print("Sent an event")
	


def handleNewImage(image_name, frame):
	#Don't send the image to event hub
	#Instead, push the image to storage and process the image with Cognitive Services
	#Then push the json from the cognitive services call to event hub
	#Then create a separate eventhub consumer client that processes each event
	#And calculates the average sentiment of the people in the photo for the last 5 minutes
	cv2.imwrite(image_name, frame)
	emotions = analyseImage(image_name, frame)
	if emotions:
		d = datetime.datetime.now()
		image_name = "face"+d
		uploadToStorage(image_name, frame)
		sendEvent(emotions+image_name)
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
 
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
newImageEvent = NewImageEvent()
newImageEvent.newImage += handleNewImage

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

    # compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

 
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
		# Send the image to Blob Storage via event queue
		#uploadToStorage("image.png", frame)
		newImageEvent.newImage("image.png", frame)
		

        # draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
eventhub_client.close()
cv2.destroyAllWindows()