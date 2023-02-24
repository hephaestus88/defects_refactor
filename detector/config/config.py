from detectron2.config import CfgNode as CN


def add_det_config(cfg, train_dataset_name, val_dataset_name, num_classes, device, output_dir):
    
    
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    # dataset
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    #workers to load data
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    
    #define learning schecule para 
    cfg.SOLVER.BASE_LR = 0.001
    
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.5
    
    #cfg.TEST_EVAL_PERIOD = 200
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes + 1

 
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = 'diou'
    cfg.MODEL_DEVICE = device
    
    cfg.OUTPUT_DIR = output_dir

    #enable AMP trainer 
    #cfg.SOLVER.AMP.ENABLED = True


    #data visualization in wandb
    cfg.USE_WANDB = True  # Comment this out if you don't want to use Wandb
    cfg.WANDB_PROJECT_NAME = "obj-det001"  # Wandb project name to log the run to
    cfg.VIS_PERIOD = 100  # Plot training results each <> iterations (sends them to Wandb)
    # # images to plot per visualization run "group", i.e. for RPN/ROI plots how many examples to show; 3 nicely fits in Wandb
    cfg.VIS_IMS_PER_GROUP = 3
    cfg.VIS_MAX_PREDS_PER_IM = 20  # Maximum number of bounding boxes per image in visualization
    cfg.VIS_TEST = True  # Visualize outputs during inference as well
    
    return cfg 