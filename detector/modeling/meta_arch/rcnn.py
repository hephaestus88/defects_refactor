"""This is the adaptation of the orginal rcnn.py file from detectron2
   This also refers to work from https://github.com/vlfom/CSD-detectron2
"""
import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from detector.utils.img_log import log_visualization_to_wandb

from detectron2.config import global_cfg
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
import detectron2.data.transforms as T
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm


@META_ARCH_REGISTRY.register()
class DetGeneralizedRCNN(GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # see if inference mode 
        if not self.training:
            return self.inference(batched_inputs)

        do_visualize = comm.is_main_process() and self.vis_period and (get_event_storage().iter % self.vis_period == 0)

        ### Preprocess inputs
        #extract the images and GTs for inputs images
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        ### Backbone feature extraction
        features = self.backbone(images.tensor)

        ### RPN proposals generation
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # visualze RPN proposals for labeled batch 
        if do_visualize:
            self._visualize_train_rpn_props(batched_inputs, proposals)

        #Visualize ROI predictions (**before** forward pass)
        if do_visualize: 
            self._visualize_train_roi_preds(batched_inputs, images, features, proposals)

        # ROI prediction
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # update loss
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        do_visualize = comm.is_main_process() and self._if_inference_visualization()

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        proposals, _ = self.proposal_generator(images, features, None)
        # visualize RPN proposals
        if do_visualize:
                self._visualize_test_rpn_props(batched_inputs, proposals)  
        
        results, _ = self.roi_heads(images, features, proposals, None)
        # visualize ROI predictions
        if do_visualize:
                self._visualize_test_roi_preds(batched_inputs, results) 
      

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def _if_inference_visualization(self):
        """Helper method that decides whether an image should be visualized in inference mode.
        The idea is to visualize a random subset of images (`cfg.VIS_IMS_PER_GROUP` in total),
        however, because inside this object we can't access all the data, all we can do is use
        a "hacky" way - flip a coin for each image, and if we haven't reached the maximum
        allowed images - visualize this one.
        """

        if not global_cfg.VIS_TEST:  # Visualization is disabled in config
            return False

        if not hasattr(self, "_vis_test_counter"):  # Lazy initialization of "images already vis-ed" counter
            self._vis_test_counter = 0

        if self._vis_test_counter >= global_cfg.VIS_IMS_PER_GROUP:  # Enough images were plotted
            return False

        # Consider visualizing each ~100th image; this heuristic would work well for datasets where
        # #images is >> `50 * cfg.VIS_IMS_PER_GROUP`
        _r = np.random.randint(50)
        if _r == 0:
            self._vis_test_counter += 1
            return True
        return False

    def _visualize_train_rpn_props(self, inputs, props):
        """Visualizes region proposals from RPN during training in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            props,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="RPN",
        )
    
    def _visualize_train_roi_preds(self, inputs, ims, feats, props):
        """Visualizes predictions from ROI head during training in Wandb. See `_visualize_predictions` for more details."""

        # First, generate bboxes predictions; for that tell ROI head that we are in the inference mode
        # so that it yield instances instead of losses, etc.
        self.roi_heads.training = False
        with torch.no_grad():  # Make sure no gradients are changed
            pred_instances, _ = self.roi_heads(ims, feats, props, None)
        self.roi_heads.training = True

        self._visualize_predictions(
            inputs,
            pred_instances,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="ROI",
        )

    def _visualize_test_rpn_props(self, inputs, props):
        """Visualizes region proposals from RPN during inference in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            props,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="RPN_test",
            title_suffix=self._vis_test_counter,
        )

    def _visualize_test_roi_preds(self, inputs, pred_instances):
        """Visualizes predictions from ROI head during inference in Wandb. See `_visualize_predictions` for more details."""

        self._visualize_predictions(
            inputs,
            pred_instances,
            viz_count=global_cfg.VIS_IMS_PER_GROUP,
            max_predictions=global_cfg.VIS_MAX_PREDS_PER_IM,
            predictions_mode="ROI_test",
            title_suffix=self._vis_test_counter,
        )


    def _visualize_predictions(
        self, batched_inputs, predictions, predictions_mode, viz_count, max_predictions, title_suffix=None
    ):
        """Visualizes images and predictions and sends them to Wandb.
        batched_inputs: List[dict], list of inputs, each must contain keys "image"
            and "instances"
        predictions: List[Instances], list of bbox instances either from RPN or ROI head,
            see `_log_visualization_to_wandb` for more details.
        predictions_mode: str, must be begin with either 'RPN' or 'ROI', defines whether
            `predictions` contain bboxes predicted by RPN or ROI heads.
        """

        assert predictions_mode[:3] in ["RPN", "ROI"], "Unsupported proposal visualization mode"

        for inp, pred in zip(batched_inputs, predictions):
            img = inp["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            gts = inp["instances"]
            suffix = f"_{viz_count}" if title_suffix is None else title_suffix
            log_visualization_to_wandb(img, gts, pred, predictions_mode, max_predictions, title_suffix=suffix)

            viz_count -= 1  # Visualize up to `viz_count` images only
            if viz_count == 0:
                break
    