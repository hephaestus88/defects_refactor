import detector.modeling.meta_arch
import detector.modeling.roi_heads
from detector.config import add_det_config
from detector.engine import DetectorTrainer
from detector.data.cocodata import *
from detector.utils.utils import on_image

from detectron2.config import get_cfg, set_global_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.engine import DefaultPredictor
import detectron2.utils.comm as comm

import os
import glob
import wandb

def setup(args):
    """setup config"""
    
    device = args.device
    num_classes = args.number  
    output_dir = args.output

    cfg = get_cfg()  # Load default configs
   
    add_det_config(
                   cfg, 
                   train_dataset_name, 
                   val_dataset_name, 
                   num_classes, 
                   device, 
                   output_dir
                   )  # Load all detection model configs
    # cfg.merge_from_file(args.config_file)  # Extend with config from specified file

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)  # Extend with config specified in args

    if args.demo_only:
        cfg.MODEL.WEIGHTS = os.path.join('output/object_detection/', 
                                 "model_final.pth")
        
    cfg.freeze()

    set_global_cfg(cfg)  # Set up "global" access for config

    default_setup(cfg, args)

    return cfg

def eval_mode(cfg):

    if cfg.VIS_TEST:
        assert cfg.USE_WANDB is True, "Plase enable your visualization"

    trainer = DetectorTrainer(cfg)
    trainer.resume_or_load(resume=False)
    res = trainer.test(cfg, trainer._trainer.model)
    if comm.is_main_process():
        verify_results(cfg, res)

    return res

def demo_test(cfg):
     """run trained model on test images and showing the test results"""
     
     predictor = DefaultPredictor(cfg)
     directory = 'test_images'
     

     if not os.path.exists(directory):
        os.makedirs(directory)
     else:
        print(f"Dirctory '{directory}'already exists")
     for n, img in enumerate(glob.glob("datasets/test/*.jpg"), 1):
        on_image(img, predictor, directory, n)

def main(args):

    #clean_cuda_cache()
    cfg = setup(args)
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if comm.is_main_process():
        if cfg.USE_WANDB:  # Set up wandb (for tracking scalars and visualizations)
            wandb.login()
            wandb.init(project=cfg.WANDB_PROJECT_NAME, config=cfg)
        else:
            assert cfg.VIS_PEROID == 0, "Visualiztion are not supported"

    if args.eval_only:  # Run evaluation
        return eval_mode(cfg)
    
    if args.demo_only:
         return demo_test(cfg)

    # setup trainer 
    trainer = DetectorTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    wandb.finish()



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Arguments:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


    