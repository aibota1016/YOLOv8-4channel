import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json


path_to_model="last.pt"
path_to_image="TFWout_3743.png"
image_url=None
threshold_bboxes=0.3
model = YOLO(path_to_model)
iou=0.1



def get_image_dimensions(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    return width, height

def denormalize_landmarks(landmarks, image_width, image_height):
    """
    Denormalize landmarks.

    Args:
    - landmarks (list): List of normalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - list: Denormalized landmarks [x1, y1, x2, y2, ..., xn, yn].
    """
    denormalized_landmarks = []

    for i in range(0, len(landmarks)):
        x_normalized, y_normalized = landmarks[i][0], landmarks[i][1]

        # Denormalize the coordinates
        x_denormalized = x_normalized * image_width
        y_denormalized = y_normalized * image_height

        denormalized_landmarks.append([x_denormalized, y_denormalized])

    return denormalized_landmarks




def test(image_path,image_url,threshold_bboxes,iou,show):

    if image_url is not None:
        response = requests.get(image_url)
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        image=image_path
        image = cv2.imread(image)

    pred_age_ev=[]
    pred_em_ev=[]
    pred_gen_ev=[]

    results = model.predict(source=image, imgsz=640,conf=threshold_bboxes,iou=iou)
    result = results[0].cpu().numpy()
    
    box=result.boxes.boxes
    bbs=[]
    face_predictions = []
    H, W, _ = image.shape
    font_size = H / 1500
    for i in range(len(box)):
        if len(box)!=0:
            x1, y1, x2, y2,confidence, label = box[i][0],box[i][1],box[i][2],box[i][3],box[i][4],box[i][5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = (box[i][2] - box[i][0])
            height = (box[i][3] - box[i][1]) 
            x_center = (box[i][0] + width/2)
            y_center = (box[i][1] + height/2)

            # Draw the bounding box on the image
            color_r = (0, 0, 255)
            color = (0, 255, 0)
            color_w = (255, 255, 255)
            thickness = int((W+H)/1300)+1  # You can change the thickness
            thickness_text = int((W+H)/2000)+1
            cv2.rectangle(image, (x1, y1), (x2, y2), color_r, thickness)
            text = f'Confidence: {confidence:.2f}'

            if label==0:
                label_t='human'
            elif label==1:
                label_t='animal'
            elif label==2:
                label_t='cartoon'

            text_label = f'Face: '+label_t


            cv2.putText(image, text_label, (x1+2, y1 - int(H/17.5)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_r, thickness_text)

            kpt=result.keypoints.data
            kp_x1, kp_y1, kp_x2, kp_y2,kp_x3, kp_y3,kp_x4, kp_y4,kp_x5, kp_y5 = kpt[i][0][0],kpt[i][0][1],kpt[i][1][0],kpt[i][1][1],kpt[i][2][0],kpt[i][2][1],kpt[i][3][0],kpt[i][3][1],kpt[i][4][0],kpt[i][4][1]
            landmarks = [(float(kp_x1), float(kp_y1)), (float(kp_x2), float(kp_y2)), (float(kp_x3), float(kp_y3)), (float(kp_x4), float(kp_y4)), (float(kp_x5), float(kp_y5))]
            landmark_colors = [(0, 255, 0),  # Green
                   (0, 0, 255),  # Red
                   (0, 255, 255),  # Yellow
                   (255, 0, 255),  # Pink
                   (255, 0, 0)] #blue
            for k, landmark in enumerate(landmarks):
                color = landmark_colors[k % len(landmark_colors)] 
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), thickness, color, -1)

            mtl=results[0].mtl
            pred_AGE=mtl[i][3:4]
            pred_age_ev.append(pred_AGE)

            if label_t=='human':
                age=str(int(mtl[i][3:4][0]))
            else:
                age="unsure"
            
            pred_GEN=mtl[i][0:3]
            
            class_GEN = np.argmax(pred_GEN.cpu())
            pred_gen_ev.append(class_GEN)
            class_labels_GEN = ['female', 'male', 'unsure']
            predicted_class_GEN = class_labels_GEN[class_GEN]

            pred_EM=mtl[i][4:]
            class_EM = np.argmax(pred_EM.cpu())
            pred_em_ev.append(class_EM)
            class_labels_EM = ['angry', 'happy', 'fear', 'sad', 'surprise', 'disgust', 'neutral','unsure']
            predicted_class_EM = class_labels_EM[class_EM]
            bbs.append([[x_center,y_center],[width,height]])
            
            cv2.putText(image, 'Emotion: '+predicted_class_EM, (x1+2, y1 - int(H/280)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_r, thickness_text)
            cv2.putText(image, 'Gender: '+predicted_class_GEN, (x1+2, y1 - int(H/25)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_r, thickness_text)
            cv2.putText(image, 'Age: '+age, (x1+2, y1 - int(H/48)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_r, thickness_text)

            face_prediction = {
                "label": label_t,
                "confidence": int(confidence*100),
                "bounding_box": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(width),
                    "height": float(height)
                },
                "facial_landmarks": landmarks,
                "emotion": predicted_class_EM,
                "gender": predicted_class_GEN,
                "age": age
            }

            face_predictions.append(face_prediction)



    with open("predictions.json", "w") as json_file:
        json.dump(face_predictions, json_file, indent=2)
    if show:
        cv2.imwrite('out.png', image)
        # Display the image with bounding box
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

test(path_to_image,image_url,threshold_bboxes,iou,show=True)



