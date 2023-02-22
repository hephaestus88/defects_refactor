import logging
import os
import weakref
from typing import Dict


from detector.checkpoint import WandbDetectionCheckpointer
from detector.utils import WandbWriter

from detectron2.data import MetadataCatalog
from detectron2.engine import (DefaultTrainer, SimpleTrainer,AMPTrainer,
                               create_ddp_model)
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.utils import comm
from detectron2.utils.events import (CommonMetricPrinter, JSONWriter,
                                     TensorboardXWriter, get_event_storage)
from detectron2.utils.logger import setup_logger



class DetectorTrainer(DefaultTrainer):
    """
    a trainer manager for detection tasks 
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = WandbDetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        raise NotImplementedError(
            "No evaluator implementation for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )


    def build_writers(self):
      
        writers = [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

        if self.cfg.USE_WANDB:
            writers.append(WandbWriter())

        return writers

