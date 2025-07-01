import os
#os.environ["OMP_NUM_THREADS"]='2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "11"
#import torch
#print(torch.cuda.device_count())  # Should return 1
#print(torch.cuda.current_device())  # Should return 0 (because it's the only available device)
#print(torch.cuda.get_device_name(0))

from ultralytics import YOLO

model = YOLO('yolov8m.yaml')  
#model = YOLO('/Users/aibota/Downloads/food_weight_pred_4channel/runs/detect/train6/weights/best.pt')

# Train the model
model.train(data='my_data.yaml', epochs=1, imgsz=640, batch=8, device="cpu", mosaic=0.0)

#model.val(data='my_data.yaml', device="cpu")
