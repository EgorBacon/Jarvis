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
 
    current_id = 0
    label_ids = {}

    @classmethod
    def load_from_folder(cls, foldername):

        for root, dirs, files in os.walk(foldername):
            for file in files:
                    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        path = os.path.join(root, file)
                        label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                        print(label, path)
                        if not label in cls.label_ids:
                            cls.label_ids[label] = cls.current_id
                            cls.current_id += 1
                        id_ = cls.label_ids[label]
                        #y_labels.append(label)#some number
                        #x_train.append(path)# verify this image, turn into a NUMPY array, GRAY
                        photo = Photo.load_from_file(path)
                        photo.label = label
                        photo.label_id = id_
                        yield photo

    @classmethod    
    def load_from_camera(cls):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        return Photo(frame)

    def __init__(self, image_data):
        self.image_data = image_data
        self.label = None
        self.label_id = None

    def detect_faces(self):
        faces = face_cascade.detectMultiScale(self.image_data, 1.5, 5)
        results = []
        for (x,y,w,h) in faces:
            roi = self.image_data[y:y+h, x:x+w]
            face = FaceROI(roi, (x,y,w,h), label = self.label, label_id = self.label_id)
            results.append(face)
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
