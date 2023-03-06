# Textile Defect Detector Tool

This repository includes the tool for fabric defect detector for my final Ph. D. project based on [Detectron2](https://github.com/facebookresearch/detectron2) framework. It also used [W&B](https://wandb.ai/) as tool to monitor the training and inference process

## Table of Contents
* [Installation](#installation)

* [Folder structure and data preparation](#folder-structure-and-data-preparation)
  * [Folder structure](#folder-structure)
  * [Image preparation](#image-preparation)

* [Run scripts](#run-scripts)
* [Configs and arguments defined](#configs-and-arguments-defined)
* [Future features](#future-features)
* [Credits](#credits)

## Installation

**Prerequisites**

If you would like to run your code locally please make sure below requirements are met:

* Linux or macOS with Python ≥ 3.7

* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. 

* OpenCV is optional but needed by demo and visualization

The code can also be run on an AWS EC2 machine. To check specifically which AMI (image) and commands needed to be run on the machine, you can this [note](https://www.notion.so/Configuring-an-AWS-EC2-VM-with-V100-GPUs-for-D2-training-shared-4c8d1487fa324aa08e4881ff3761d121) as a reference.

**Installing Detectron2**

See [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) from Detectron2's tutorial.


## Folder structure and data preparation

### Folder structure
**Directory layout for fabric defect detection tool's folders**
>  

    .
    ├── configs                 # config files for model building
    ├── datasets                # datasets of the target images
    │   ├── train               # folder for storing images and annotations for train
    │   ├── test                # folder for storing images and annotations for test
    │   └── valid               # folder for storing images and annotations for validation
    |── detector                # folder for detector(model)
    |   └── checkpoint          # folder to store script for save ckpt to wandb
    |   └── config              # folder to store script for model configuration(parameters)
    |   └── data                # folder to store script for coco data registration
    |   └── engine              # folder to store script for model trainer(engine)
    |   └── modeling            # folder to store script for custom model components
    |   └── utils               # scripts to log to the wandb and support functions
    |── output                  # output for model weights and train log
    |── wandb                   # folder to store files in wandb
>

### Image preparation

Step1: Download your images and annotations files(in coco format)

Step2: Save the images and annotations files in three separate folder `train`, `test`, `valid` under the `datasets` folder.

Step3: code in the folder `detector/data` can help register the data

## Running scripts

1. If you want to train the model using default model config parameters and arguments, you can just run 

    ```bash
     # train the model
     $ python train_net.py 
     ``` 
2. If you want to train the model using own arguments, you can run as below. more details about the parameters and arguments will be introduced in the [following section](#configs-and-arguments-defined)
    ```bash
     # --num-gpus for the number of gpus used in training
     # --config-file for the configs file used for model
     $ python train_net.py --num-gpus 1 --config-file configs/coco/default_faster_rcnn_R_50_FPN_3x.yaml
     ``` 
3. If you want to run the evaluation, you can run the following(`MODEL_WEIGHTS` are the target weights saved in checkpoit output folder):
    ```bash
     # --run the evaluation 
     $ python run_net.py --eval-only MODEL.WEIGHTS output/object_detection/model_final.pth
     ```
4. If you want to resume the training, you can run the following with initial mdoel weights you expected to set:
    ```bash
     # --resume the training
     $ python run_net.py --resume MODEL.WEIGHTS output/object_detection/model_0000199.pth
    ```
5. If you want to displayed results of the trained model on test image datasets, you can run the following:
    ```bash
     # --run the demo
     $ python run_net.py --demo-only 
    ```

## Configs and arguments defined
### Configurations 
**Here listed partial configurations, you can see more details in [detectron2 doc](https://detectron2.readthedocs.io/en/latest/)**
Configure parameters                 |    Parameters 
-------------------------------------|-------------
DATASETS.TRAIN                       |    define dataset name for training
DATASETS.TEST                        |    define dataset name for test
DATALOADER.NUM_WORKERS               |    define number of data loading threads
SOLVER.IMS_PER_BATCH                 |    define the image number per batch 
SOLVER.BASE_LR                       |    define the base learning rate 
SOLVER.WARMUP_ITERS                  |    define the
SOLVER.WEIGHT_DECAY                  |    define weight decay 
SOLVER.STEPS                         |    define the interation number to decrease learning rate by GAMMA
SOLVER.GAMMA                         |    define the value of the GAMMA
MODEL.META_ARCHITECTURE              |    Set model's meta_architecture
MDOEL.PROPOSAL_GENERATOR.NAME        |    Set model's RPN network 
MODEL.ROI_HEADS.NUM_CLASSES          |    Define the foreground class number 
MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE|    Define the reg loss for head bounding box
USE_WANDB                            |    Enable using the WANDB


### Arguments for training 
Parameters              |    Description 
------------------------|--------------------------
-num-gpus               |     number of gpus used for training per **machine** default=1,
--num-machines          |     default=1, total number of machines
--config-file           |     configure file default=""
MODEL.WEIGHTS           |     model weights  default="/path/to/weight.pth "
--resume                |     Whether to attempt to resume from the checkpoint directory.
-eval-only              |     action="store_true", "perform evaluation only"
--demo-only             |     action="store_true", "perform demo on test imgs only"
--device                |     devices used for training: ["cuda", "cpu"], default='cuda'
--number                |     number of detection classes(w/o) background, default=3    
--output                |     output for the model training, default='./output/object_detection'


## Future features

- [ ] Add other meta-arch stuctur
- [x] Add channel based attention mechanism to the resnet


## Credits

