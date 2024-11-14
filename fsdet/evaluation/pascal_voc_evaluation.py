# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
import os
import datetime
import matplotlib.pyplot as plt
from fsdet.data import MetadataCatalog
from fsdet.utils import comm
from fsdet.utils.logger import create_small_table

from .evaluator import DatasetEvaluator


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the datasets, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        try:
            self._base_classes = meta.base_classes
            self._novel_classes = meta.novel_classes
        except AttributeError:
            self._base_classes = meta.thing_classes
            self._novel_classes = None
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            recalls = defaultdict(list) # iou -> recall per class
            recalls_base = defaultdict(list)  # iou -> recall per class for base classes
            recalls_novel = defaultdict(list)  # iou -> recall per class for novel classes
            precs = defaultdict(list)
            exist_base, exist_novel = False, False
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)
                    recalls[thresh].append(rec * 100)  # Store recall values
                    precs[thresh].append(prec*100)
                    
                    if self._base_classes is not None and cls_name in self._base_classes:
                        aps_base[thresh].append(ap * 100)
                        recalls_base[thresh].append(rec * 100)
                        exist_base = True

                    if self._novel_classes is not None and cls_name in self._novel_classes:
                        aps_novel[thresh].append(ap * 100)
                        recalls_novel[thresh].append(rec * 100)
                        exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        mRecall = {iou: np.mean(x) for iou, x in recalls.items()}  # Calculate mean recall
        mF1 = {iou: 2 * (mAP[iou] * mRecall[iou]) / (mAP[iou] + mRecall[iou]) for iou in aps.keys()}
        ret["bbox"] = {
                        "AP": np.mean(list(mAP.values())),
                        "AP50": mAP[50],
                        "AP75": mAP[75],
                        "Recall": np.mean(list(mRecall.values())),
                        "Recall50": mRecall[50],
                        "Recall75": mRecall[75],
                        "F1": np.mean(list(mF1.values())),  # 添加 F1-score 的平均值
                        "F150": mF1[50],                    # 添加 50 阈值下的 F1-score
                        "F175": mF1[75]                     # 添加 75 阈值下的 F1-score
                         }
        # ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75],
        #                "Recall": np.mean(list(mRecall.values())), "Recall50": mRecall[50], "Recall75": mRecall[75]}

        # adding evaluation of the base and novel classes
        # if exist_base:
        #     mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
        #     ret["bbox"].update(
        #         {"bAP": np.mean(list(mAP_base.values())), "bAP50": mAP_base[50],
        #          "bAP75": mAP_base[75]}
        #     )

        # if exist_novel:
        #     mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
        #     ret["bbox"].update({
        #         "nAP": np.mean(list(mAP_novel.values())), "nAP50": mAP_novel[50],
        #         "nAP75": mAP_novel[75]
        #     })
        
                # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            mRecall_base = {iou: np.mean(x) for iou, x in recalls_base.items()}  # Calculate recall for base classes
            mF1 = {iou: 2 * (mAP[iou] * mRecall[iou]) / (mAP[iou] + mRecall[iou]) for iou in aps.keys()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())), 
                 "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75],
                 "bRecall": np.mean(list(mRecall_base.values())),  # Overall recall for base classes
                 "bRecall50": mRecall_base[50],  # Recall at IoU 50 for base classes
                 "bRecall75": mRecall_base[75],   # Recall at IoU 75 for base classes
                "bF1": np.mean(list(mF1.values())),  # 添加 F1-score 的平均值
                "bF150": mF1[50],                    # 添加 50 阈值下的 F1-score
                "bF175": mF1[75]                   
                 
                 }
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            mRecall_novel = {iou: np.mean(x) for iou, x in recalls_novel.items()}  # Calculate recall for novel classes
            mF1 = {iou: 2 * (mAP_novel[iou] * mRecall_novel[iou]) / (mAP_novel[iou] + mRecall_novel[iou]) for iou in aps_novel.keys()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())), 
                "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75],
                "nRecall": np.mean(list(mRecall_novel.values())),  # Overall recall for novel classes
                "nRecall50": mRecall_novel[50],  # Recall at IoU 50 for novel classes
                "nRecall75": mRecall_novel[75],   # Recall at IoU 75 for novel classes
                "nF1": np.mean(list(mF1.values())),  # 添加 F1-score 的平均值
                "nF150": mF1[50],                    # 添加 50 阈值下的 F1-score
                "nF175": mF1[75]  
            })

        # write per class recall and AP to logger
        per_class_res = {self._class_names[idx]: (ap) for idx, (ap) in enumerate(zip(aps[50]))}
        self._logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        current_time = datetime.datetime.now().strftime("%Y%m%d")
        
        save_dir = os.path.join("./ksh", f"evaluation_{current_time}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 仅选择关键的 IoU 阈值来绘制曲线
        # selected_iou_thresholds = [50, 75, 85]

        # 绘制每个 IoU 阈值的 P-R 曲线
        plt.figure(figsize=(10, 8))
        for iou in range(50, 100, 5):
            if iou in recalls and iou in precs:
                # rec_a = np.mean(recalls[iou][0])
                # ap_a = np.mean(precs[iou][0])
                
                # 获取当前 IoU 阈值下的 Recall 和 Precision 列表
                recalls[iou][0] = recalls[iou][0][::10]
                rec_values = recalls[iou]
                precs[iou][0] = precs[iou][0][::10]
                prec_values = precs[iou]
                # plt.plot(rec_a,ap_a, label=f'IoU={iou}%', marker='o')
                # 绘制 P-R 曲线
                plt.plot(rec_values[0], prec_values[0], label=f'IoU={iou}%', marker='o',markersize=1)

        # 设置图形标签和标题
        plt.xlim(0, 100)  # Recall 的范围从 0 到 100
        plt.ylim(0, 100)  # Precision 的范围从 0 到 100
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout()
        plt.grid(True)

        # 保存 P-R 曲线图像
        sj = datetime.datetime.now().strftime("%H%M%S")
        NAME = "st"
        pr_curve_path = os.path.join(save_dir, sj+NAME+"precision_recall_curve.png")
        plt.savefig(pr_curve_path, format="png", bbox_inches="tight")
        plt.close()  # 关闭图像，释放内存

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}
        # tishifus = ("attn_box:",attn_box,"---"*3,"attn_fused:",attn_fused)
        # self._logger.info(tishifus)
        # self._logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n"+create_small_table(ret["bbox"]))
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""

                
                
@lru_cache(maxsize=None)
def parse_rec(filename):
    
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        if obj_struct["name"] == "none":
            continue
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects




def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap