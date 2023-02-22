import detector.modeling.meta_arch
import detector.modeling.roi_heads
import detectron2.data
import detectron2.utils.comm as comm
import wandb
from detector.config import add_det_config
from detector.engine import DetectorTrainer
from detectron2.config import get_cfg, set_global_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.data.datasets import register_coco_instances



import os
import pickle
#output directory for obj det model
output_dir = "./output/object_detection"
num_classes = 3

device =  "cuda" # "cuda" or "cpu"

train_dataset_name = "defect_train"
train_images_path = "datasets/train"
train_json_annot_path = "./datasets/train/_annotations.coco.json"

val_dataset_name = "defect_val"
val_images_path = "datasets/valid"
val_json_annot_path = "./datasets/valid/_annotations.coco.json"

test_dataset_name = "defect_test"
test_images_path = "datasets/test"
test_json_annot_path = "./datasets/test/_annotations.coco.json"


cfg_save_path = "OD_cfg.pickle"

def setup(args):
    """setup config"""

    cfg = get_cfg()  # Load default configs
    add_det_config(cfg, train_dataset_name, 
                   val_dataset_name, 
                   num_classes, 
                   device, 
                   output_dir)  # Load all detection model configs
    cfg.merge_from_file(args.config_file)  # Extend with config from specified file
    cfg.merge_from_list(args.opts)  # Extend with config specified in args


    print(cfg.dump())
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

def main(args):

    cfg = setup(args)

    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if comm.is_main_process():
        if cfg.USE_WANDB:  # Set up wandb (for tracking scalars and visualizations)
            wandb.login()
            wandb.init(project=cfg.WANDB_PROJECT_NAME, config=cfg)
        else:
            assert cfg.VIS_PEROID == 0, "Visualiztion are not supported"

    if args.eval_only:  # Run evaluation
        return eval_mode(cfg)

    # setup trainer 
    trainer = DetectorTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )