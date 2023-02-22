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



def setup(args):
    """setup config"""

    cfg = get_cfg()  # Load default configs
    add_det_config(cfg)  # Load all detection model configs
    cfg.merge_from_file(args.config_file)  # Extend with config from specified file
    cfg.merge_from_list(args.opts)  # Extend with config specified in args

    print(cfg.dump())
    cfg.freeze()

    set_global_cfg(cfg)  # Set up "global" access for config

    default_setup(cfg, args)

    return cfg

def eval_mode(cfg):

    trainer = DetectorTrainer(cfg)
    trainer.resume_or_load(resume=False)
    res = trainer.test(cfg, trainer._trainer.model)
    if comm.is_main_process():
        verify_results(cfg, res)

    return res

def main(args):

    cfg = setup(args)

    if comm.is_main_process():
        if cfg.USE_WANDB:  # Set up wandb (for tracking scalars and visualizations)
            wandb.login()
            wandb.init(project=cfg.WANDB_PROJECT_NAME, config=cfg)
       

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