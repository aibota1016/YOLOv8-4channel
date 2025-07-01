import os
os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
import os
import glob
from PIL import Image


# Load a model
model = YOLO('/workspace/results/rgb_9k_weight_138_classes_new_split/weights/best.pt')  # build a new model from YAML

image_folder = "/workspace/Food_photos"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

out_folder = '/workspace/Model_preds_for_experiment'

#results = model.predict(data='data_9k_gfsd.yaml', imgsz=640, device=5, split='test', classes=[], save_txt=True)

#results = model.predict(image_files, imgsz=640, device=0, conf=0.25, project=out_folder, save=True)


for img in os.listdir(image_folder):
    #results = model.predict(os.path.join(image_folder, img), imgsz=640, device=3, show=False, save=False, save_txt=True, project=out_folder, name='', exist_ok=True)
    img_path = os.path.join(image_folder, img)
    results = model.predict(img_path, imgsz=640, device=3, show=False, save=False, save_txt=False)
    
    txt_save_path = os.path.join(out_folder, os.path.splitext(img)[0] + ".txt")
    #txt_save_path = os.path.join(out_folder, img.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt').replace('.JPG', '.txt'))
    with open(txt_save_path, "w") as f:
        for i, pred in enumerate(results[0].boxes.data):
            #class_id, x_center, y_center, width, height, confidence = pred.tolist()
            class_id = int(pred[5].item()) 
            x_center, y_center, width, height, confidence = pred[:5].tolist()
            
            mtl = results[0].mtl
            pred_weight = mtl[i][2].item()  # Extract custom data

            # Save in YOLO format with the additional custom value
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height} {confidence} {pred_weight}\n")

print("Inference and saving completed.")

"""
for i, (r, img_path) in enumerate(zip(results, image_files)):
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    
    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    r.save(filename=os.path.join(out_folder, f"{base_filename}.jpg"))
"""
