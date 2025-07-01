import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests


path_to_model="last.pt"
path_to_image="ex/RAFdb_test_0003.jpg"
image_url=None
threshold_bboxes=0.3
model = YOLO(path_to_model)



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



def test(video_source=0, threshold_bboxes=0.3, show=True):
    
    while True:
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        small_frame = cv2.resize(frame, (640, 480))

            # Perform the detection and analysis on the frame
        results = model.predict(source=small_frame, imgsz=640, conf=threshold_bboxes)
        result = results[0].cpu().numpy()
        
        box=result.boxes.boxes


        for i in range(len(box)):
            if len(box)!=0:
                x1, y1, x2, y2,confidence, label = box[i][0],box[i][1],box[i][2],box[i][3],box[i][4],box[i][5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = (box[i][2] - box[i][0])
                height = (box[i][3] - box[i][1]) 

                # Draw the bounding box on the image
                color = (0, 255, 0)  # You can change the color (BGR format)
                thickness = 2  # You can change the thickness
                cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, thickness)
                text = f'Confidence: {confidence:.2f}'

                if label==0:
                    label_t='human'
                elif label==1:
                    label_t='animal'
                elif label==2:
                    label_t='cartoon'

                text_label = f'Label: '+label_t

                cv2.putText(small_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(small_frame, text_label, (x1, y1 - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                kpt=result.keypoints.data
                kp_x1, kp_y1, kp_x2, kp_y2,kp_x3, kp_y3,kp_x4, kp_y4,kp_x5, kp_y5 = kpt[i][0][0],kpt[i][0][1],kpt[i][1][0],kpt[i][1][1],kpt[i][2][0],kpt[i][2][1],kpt[i][3][0],kpt[i][3][1],kpt[i][4][0],kpt[i][4][1]
                landmarks = [(float(kp_x1), float(kp_y1)), (float(kp_x2), float(kp_y2)), (float(kp_x3), float(kp_y3)), (float(kp_x4), float(kp_y4)), (float(kp_x5), float(kp_y5))]
                color = (0, 255, 0) 
                for landmark in landmarks:
                    cv2.circle(small_frame, (int(landmark[0]), int(landmark[1])), 3, color, -1)

                mtl=results[0].mtl


                if label_t=='human':
                    age=str(int(mtl[i][3:4][0]))
                else:
                    age="unsure"
                
                pred_GEN=mtl[i][0:3]
                
                class_GEN = np.argmax(pred_GEN.cpu())
                class_labels_GEN = ['female', 'male', 'unsure']
                predicted_class_GEN = class_labels_GEN[class_GEN]

                pred_EM=mtl[i][4:]
                class_EM = np.argmax(pred_EM.cpu())
                class_labels_EM = ['angry', 'happy', 'fear', 'sad', 'surprise', 'disgust', 'neutral','unsure']
                predicted_class_EM = class_labels_EM[class_EM]
                cv2.putText(small_frame, 'Emotion: '+predicted_class_EM, (x1, y1 - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(small_frame, 'Gender: '+predicted_class_GEN, (x1, y1 - 49), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(small_frame, 'Age: '+age, (x1, y1 - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if show:
            # Display the frame with bounding boxes
            cv2.imshow('Frame with Bounding Boxes', small_frame)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Release the capture object and close all windows
        cap.release()



test(video_source=0, threshold_bboxes=threshold_bboxes, show=True)



