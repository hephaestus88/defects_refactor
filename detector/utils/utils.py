from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from detectron2.utils.visualizer import ColorMode

import random
import cv2
import matplotlib.pyplot as plt
import torch
import gc
import os


def plot_samples(dataset_name, n = 2):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for d in random.sample(dataset_custom, n):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], 
                                metadata=dataset_custom_metadata, 
                                scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(15, 20))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()
        
    
    
def clean_cuda_cache():
        torch.cuda.empty_cache()
        gc.collect()



# define a function to run instance on image        

def on_image(image_path, predictor, path, n):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1],
                   metadata={"thing_classes":['defects','misprint','oil','seam']},
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save(os.path.join(path, 'img_{}.jpg'.format(n)))

    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()
    

# define a function to run instance on video
def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()==False):
        print("Error when opening video file...")
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:,:,::-1],
                   metadata={},
                   scale=0.5,
                   instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()
    

