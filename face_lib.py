import cv2
import os
from PIL import Image
import numpy as np

script_dir = os.path.dirname(__file__)
classifier_path = os.path.join(script_dir, './data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(classifier_path)
recognniser = cv2.face.LBPHFaceRecognizer_create()

class Photo(object):
    @classmethod
    def load_from_file(cls, filename):
        pil_image = Image.open(filename).convert("L")
        image_data = np.array(pil_image)
        return Photo(image_data)

    @classmethod    
    def load_from_camera(cls):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        return Photo(frame)

    def __init__(self, image_data):
        self.image_data = image_data

    def detect_faces(self):
        faces = face_cascade.detectMultiScale(self.image_data, 1.5, 5)
        results = []
        for (x,y,w,h) in faces:
            roi = self.image_data[y:y+h, x:x+w]
            results.append(FaceROI(roi, (x,y,w,h)))
        return results

class FaceROI(object):
    def __init__(self, image_data, coordinates, label = None, label_id=None):
        self.image_data = image_data
        self.coordinates = coordinates
        self.label = label
        self.label_id = label_id

    def train(self, database, label):
        pass

    def classify(self, database):
        pass
