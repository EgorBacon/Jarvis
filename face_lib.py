import cv2
import os

class Photo(object):
    @classmethod
    def load_from_file(cls, filename):
        pass

    @classmethod    
    def load_from_camera(cls):
        pass

    def __init__(self, image_data):
        self.image_data = image_data

    def detect_faces(self):
        faces = [FaceROI(d) for d in ...]
        return faces


class FaceROI(object):
    def __init__(self, image_data):
        self.image_data = image_data

    def train(self, database):
        pass

    def classify(self, database):
        pass
