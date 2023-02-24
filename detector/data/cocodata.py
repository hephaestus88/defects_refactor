from detectron2.data.datasets import register_coco_instances

train_dataset_name = "defect_train"
train_images_path = "datasets/train"
train_json_annot_path = "./datasets/train/_annotations.coco.json"

val_dataset_name = "defect_val"
val_images_path = "datasets/valid"
val_json_annot_path = "./datasets/valid/_annotations.coco.json"

test_dataset_name = "defect_test"
test_images_path = "datasets/test"
test_json_annot_path = "./datasets/test/_annotations.coco.json"

#register train dataset
register_coco_instances(name = train_dataset_name, 
                        metadata={}, 
                        json_file=train_json_annot_path, 
                        image_root=train_images_path)

#register validation dataset
register_coco_instances(name = val_dataset_name, 
                        metadata={}, 
                        json_file=val_json_annot_path, 
                        image_root=val_images_path)

#register test dataset
register_coco_instances(name = test_dataset_name, 
                        metadata={}, 
                        json_file=test_json_annot_path, 
                        image_root=test_images_path)