from ultralytics import YOLO
import cv2

path_to_model="last.pt"
model = YOLO(path_to_model)
results=model.predict(source="0",show=True)
print("hello")