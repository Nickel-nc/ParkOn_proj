from pathlib import Path
import mrcnn.config
import mrcnn.utils
import numpy as np
import os

# Routes configure:
with open('access.txt', 'r') as f:
    split = f.read().split()
    RTSP_SOURCE = split[0]
    MONGO_SOURCE = split[1]

# Root directory of the project
ROOT_DIR = Path(".")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Video file or camera to process - set this to 0 to use your webcam instead of a video file
LOCAL_VIDEO_SOURCE = "test_images/test_5.mp4"
# Frame Per Second (FPS) - skips frames
FRAME_RATE = 30
# Video file output destination
OUTPUT_SOURCE = "output/output_TEST.avi"
OUTPUT_PARAMS = 0

# Set 1 to write images for inspection
WRITE_IMG_LOGS = True
STORED_FRAMES_DIR = "serv_snaps"
# Manually setting coordinates for parking area
SHOW_PARKING_AREA = False
CALC_PARK_AREA = False
PARKING_AREA_PTS = np.array([[620,785], [1320,890], [1345,945], [1134,933], [1110,985], [575,875]])

# Period for uploading results to Mongo server in seconds
SEEK_RATE = 30
FRAME_SIZE = (1080, 1920)
OVERLAP_THRESH_HOLD = 0.15

# Mongo Settings:

WRITE_TO_DB_MODE = True
REWRITE_DB = False  # clear collection for every connection
CAM_ID = "5ed27b0f8c07852e00cf1a90"
DATABASE_NAME = "parkonDb"  # collection nn_output

# Plots area where model is working
SHOW_DETECTION_FRAME = False
# top-left and bottom-right detection coordinates
X1 = 500
Y1 = 780
X2 = 1350
Y2 = 970


INIT_PARKING_ZONES = np.array([[  0,  22,  96, 137],
       [  4, 146, 109, 248],
       [ 20, 252, 123, 356],
       [ 43, 365, 147, 463],
       [ 45, 475, 145, 560],
       [ 60, 570, 124, 662],
       [ 64, 671, 137, 779]])

INIT_IDS = np.array([8,  3,  3,  3,  3,  8])

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.4

