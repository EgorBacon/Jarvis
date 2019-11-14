import os
import cv2
import numpy as np
import pickle
from face_lib import Photo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, './data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(classifier_path)
recognniser = cv2.face.LBPHFaceRecognizer_create()

dataset = []

for photo in Photo.load_from_folder(img_dir):
	faces = photo.detect_faces()
	print(str(len(faces)) + " faces detected")
	for face in faces:
		dataset.append(face)

#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as pf:
	pickle.dump(Photo.label_ids, pf)

labels = [f.label_id for f in dataset]
train = [f.image_data for f in dataset]


recognniser.train(train, np.array(labels))
recognniser.save("trainer.yml")
