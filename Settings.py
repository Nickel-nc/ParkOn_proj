from pathlib import Path
import mrcnn.config
import mrcnn.utils
import numpy as np
import os

# Routes configure:
with open('access.txt', 'rb') as f:
    RTSP_SOURCE = f.read().split()[0]
    MONGO_SOURCE = f.read().split()[1]

# Root directory of the project
ROOT_DIR = Path(".")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Video file or camera to process - set this to 0 to use your webcam instead of a video file
LOCAL_VIDEO_SOURCE = "test_images/test3.avi"
# Frame Per Second (FPS) - skips frames
FRAME_RATE = 3
# Video file output destination
OUTPUT_SOURCE = "output/output_TEST.avi"
OUTPUT_PARAMS = 0
# Manually setting coordinates for parking area
SHOW_PARKING_AREA = False
PARKING_AREA_PTS = np.array([[620,785], [1320,890], [1345,945], [1134,933], [1110,985], [575,875]])


# SKIP_FRAMES = 100
FRAME_SIZE = (1080, 1920)
OVERLAP_THRESH_HOLD = 0.15

# Mongo Settings:

WRITE_TO_DB_MODE = False
LAT_LON_COORDS = []
CAM_ID = '01'
DATABASE_NAME = "parkonDb"  # collection nn_output

# Plots area where model is working
SHOW_DETECTION_FRAME = False
# top-left and bottom-right detection coordinates
X1 = 575
Y1 = 780
X2 = 1350
Y2 = 1050

INIT_PARKING_ZONES = np.array([[83, 634, 159, 741],
                               [72, 552, 159, 630],
                               [69, 467, 180, 549],
                               [59, 358, 167, 442],
                               [32, 239, 130, 327],
                               [0, 0, 93, 99],
                               [17, 113, 124, 215]])

INIT_IDS = np.array([8,  3,  3,  3,  3,  8])

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.25

