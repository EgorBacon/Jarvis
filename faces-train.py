import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, './data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(classifier_path)
recognniser = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


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
				pil_image = Image.open(path).convert("L")
				size = (500,500)
				final_img = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_img, "uint8")
				print(image_array)
				faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)

#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognniser.train(x_train, np.array(y_labels))
recognniser.save("trainer.yml")
