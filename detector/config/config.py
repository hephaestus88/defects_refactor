from detectron2.config import CfgNode as CN


def add_det_config(cfg):
    
    
    cfg.MODEL.META_ARCHITECTURE = ""
    cfg.MODEL.ROI_HEADS.NAME = ""
    

    return cfg 