
import argparse
import numpy as np
import os
from itertools import chain
import cv2
from PIL import Image

from fsdet.config import get_cfg
from fsdet.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from fsdet.data import detection_utils as utils
from fsdet.utils.logger import setup_logger
from fsdet.utils.visualizer import Visualizer

from fsdet.data.dataset_mapper import AlbumentationMapper

def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            
            if os.path.exists(filepath):
                filepath = filepath[:-4] + '_dup_' + str(np.random.randint(0, 1000)) + '.jpg'
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 2.0 if args.show else 1.0
    if args.source == "dataloader":
        mapper = None
        if cfg.INPUT.USE_ALBUMENTATIONS:
            mapper = AlbumentationMapper(cfg, is_train=True)
        train_data_loader = build_detection_train_loader(cfg, mapper=mapper)
        for batch in train_data_loader:
            for per_image in batch:
                
                

                
                img = per_image["image"].permute(1, 2, 0)
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                )

                
                num_instances = len(per_image['instances'])
                output(vis, "I{}_".format(num_instances) + str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        for dic in dicts:
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
