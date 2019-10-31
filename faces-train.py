import os
import cv2
import numpy as np
from PIL import Image
import pickle
from face_lib import FaceROI, Photo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, './data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(classifier_path)
recognniser = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

dataset = []

for root, dirs, files in os.walk(img_dir):
	for file in files:
			if file.endswith(".png") or file.endswith(".jpg"):
				path = os.path.join(root, file)
				label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
				print(label, path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				print(label_ids)
				#y_labels.append(label)#some number
				#x_train.append(path)# verify this image, turn into a NUMPY array, GRAY
				photo = Photo.load_from_file(path)
				faces = photo.detect_faces()
				for face in faces:
					face.label = label
					face.label_id = id_
					dataset.append(face)

#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

labels = [f.label_id for f in dataset]
train = [f.image_data for f in dataset]


recognniser.train(train, np.array(labels))
recognniser.save("trainer.yml")
