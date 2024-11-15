
from .config import CfgNode as CN















_C = CN()

_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"



_C.MODEL.WEIGHTS = ""




_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]



_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]





_C.INPUT = CN()

_C.INPUT.MIN_SIZE_TRAIN = (800,)


_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

_C.INPUT.MAX_SIZE_TRAIN = 1333

_C.INPUT.MIN_SIZE_TEST = 800

_C.INPUT.MAX_SIZE_TEST = 1333


_C.INPUT.CROP = CN({"ENABLED": False})





_C.INPUT.CROP.TYPE = "relative_range"


_C.INPUT.CROP.SIZE = [0.9, 0.9]






_C.INPUT.FORMAT = "BGR"

_C.INPUT.VIS_PERIOD = 1000





_C.DATASETS = CN()

_C.DATASETS.TRAIN = ()

_C.DATASETS.TEST = ()




_C.DATALOADER = CN()

_C.DATALOADER.NUM_WORKERS = 4



_C.DATALOADER.ASPECT_RATIO_GROUPING = True

_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

_C.DATALOADER.REPEAT_THRESHOLD = 0.0


_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True




_C.MODEL.BACKBONE = CN()


_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"


_C.MODEL.BACKBONE.FREEZE_AT = 2

_C.MODEL.BACKBONE.FREEZE = False
_C.MODEL.BACKBONE.FREEZE_P5 = False




_C.MODEL.FPN = CN()



_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256


_C.MODEL.FPN.NORM = ""


_C.MODEL.FPN.FUSE_TYPE = "sum"





_C.MODEL.PROPOSAL_GENERATOR = CN()

_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"


_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0

_C.MODEL.PROPOSAL_GENERATOR.FREEZE = False





_C.MODEL.ANCHOR_GENERATOR = CN()

_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"





_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]





_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]



_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]





_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  



_C.MODEL.RPN.IN_FEATURES = ["res4"]


_C.MODEL.RPN.BOUNDARY_THRESH = -1









_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]

_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256

_C.MODEL.RPN.POSITIVE_FRACTION = 0.5

_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0


_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000






_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

_C.MODEL.RPN.NMS_THRESH = 0.7




_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"

_C.MODEL.ROI_HEADS.NUM_CLASSES = 80



_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]



_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]




_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25


_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True



_C.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
_C.MODEL.ROI_HEADS.FREEZE_FEAT = False

_C.MODEL.ROI_HEADS.UNFREEZE_FC2 = False

_C.MODEL.ROI_HEADS.UNFREEZE_FC1 = False



_C.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
_C.MODEL.ROI_HEADS.COSINE_MARGIN = 0.35










_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05


_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5





_C.MODEL.ROI_BOX_HEAD = CN()


_C.MODEL.ROI_BOX_HEAD.NAME = ""


_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14

_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0

_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0

_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0

_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256


_C.MODEL.ROI_BOX_HEAD.NORM = ""

_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False

_C.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT = 1.0
_C.MODEL.ROI_BOX_HEAD.BOX_CLS_WEIGHT = 1.0


_C.MODEL.ROI_BOX_HEAD.SUB_FC_DIM = 1024


_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH = CN({'ENABLED': False})  
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM = 128
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE = 0.1
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY = CN({'ENABLED': False})
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS = [8000, 16000]
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE = 0.2
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD = 0.5
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION = 'V1'
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC = 'none'

_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE = CN({'ENABLED': False})
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE.START = 3000
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE.IOU_THRESHOLD = 0.8


_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE = CN()  
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.DATASET = 'PASCAL VOC'
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.PATH = ''
_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.DISABLE_PROTOTYPE_GRAD = True

_C.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY = False








_C.MODEL.MOCO = CN({'ENABLED': False})

_C.MODEL.MOCO.DEBUG_DEQUE_AND_ENQUE = False

_C.MODEL.MOCO.SAVE_QUEUE_ITERS = 0

_C.MODEL.MOCO.MOMENTUM = 0.99
_C.MODEL.MOCO.QUEUE_SIZE = 65536
_C.MODEL.MOCO.TEMPERATURE = 0.1
_C.MODEL.MOCO.MLP_DIMS = [1024, 128]

_C.MODEL.MOCO.WARM_UP_STEPS = 200

_C.MODEL.MOCO.CLS_LOSS_WEIGHT = 1.0
_C.MODEL.MOCO.MOCO_LOSS_WEIGHT = 1.0






_C.MODEL.RETINANET = CN()


_C.MODEL.RETINANET.NUM_CLASSES = 80

_C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]



_C.MODEL.RETINANET.NUM_CONVS = 4





_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]




_C.MODEL.RETINANET.PRIOR_PROB = 0.01



_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5


_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1







_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  


_C.MODEL.RESNETS.NUM_GROUPS = 1


_C.MODEL.RESNETS.NORM = "FrozenBN"



_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64



_C.MODEL.RESNETS.STRIDE_IN_1X1 = True


_C.MODEL.RESNETS.RES5_DILATION = 1


_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64



_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]


_C.MODEL.RESNETS.DEFORM_MODULATED = False

_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1





_C.MODEL.DLA = CN()

_C.MODEL.DLA.OUT_FEATURES = ['level2', 'level3', 'level4', 'level5']



_C.MODEL.DLA.ARCH = 'DLA-34'





_C.SOLVER = CN()

_C.SOLVER.NAME = 'SGD'
_C.SOLVER.MASKED_PARAMS = []
_C.SOLVER.MASKED_PARAMS_INDS = []


_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001


_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000




_C.SOLVER.IMS_PER_BATCH = 16





_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY




_C.TEST = CN()



_C.TEST.EXPECTED_RESULTS = []


_C.TEST.EVAL_PERIOD = 0


_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200







_C.TEST.KEYPOINT_OKS_SIGMAS = []

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True






_C.OUTPUT_DIR = "./output"



_C.SEED = -1




_C.CUDNN_BENCHMARK = False


_C.MUTE_HEADER = True









_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0






_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False







_C.INPUT.MASK_FORMAT = "polygon"  







_C.DATASETS.PROPOSAL_FILES_TRAIN = ()

_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000


_C.DATASETS.PROPOSAL_FILES_TEST = ()

_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000





_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()

_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)





_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256


_C.MODEL.ROI_MASK_HEAD.NORM = ""

_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False

_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"





_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  


_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1














_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True




_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"




_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]


_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255

_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54

_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128

_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4

_C.MODEL.SEM_SEG_HEAD.NORM = "GN"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN = CN()

_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0


_C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5









_C.INPUT.USE_TRANSFORM_AUG = False


    
    

_C.INPUT.USE_ALBUMENTATIONS = False
_C.INPUT.ALBUMENTATIONS_JSON = '/data/few-shot-det/albumentation/detection-albu-config-VOC.json'

