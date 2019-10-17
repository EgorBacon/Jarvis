import numpy as np
import cv2
import pickle
import os

script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, './data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(classifier_path)
recognniser = cv2.face.LBPHFaceRecognizer_create()
labels = {"person's name": 1}
recognniser.read("trainer.yml")
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
print(labels)

cap = cv2.VideoCapture(0)

while(True):
	#capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.5, 5)

	for (x,y,w,h) in faces:
		#finding where the face is
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		#recognize? Deep learning model to predict things
		id_, conf = recognniser.predict(roi_gray)
		print(id_, conf)
		if conf >= 50: #and conf<=60:
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (0, 255, 0)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		else:
			print("No common faces detected")

		img_item = "my-image.png"
		cv2.imwrite(img_item, roi_gray)

		#making the square around the face
		color = (0, 255, 0) #BRG
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)

	#display result frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows
