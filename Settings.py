from pathlib import Path
import mrcnn.config
import mrcnn.utils
import os

# Root directory of the project
ROOT_DIR = Path(".")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Video file or camera to process - set this to 0 to use your webcam instead of a video file
LOCAL_VIDEO_SOURCE = "test_images/test2.mp4"
RTSP_SOURCE = "rtsp://89.179.245.17:554/user=admin_password=tlJwpbo6_channel=0_stream=0.sdp"
# Video file output destination
OUTPUT_SOURCE = "output/output.mp4"
OUTPUT_PARAMS = 0

# Manual setting coordinates for parking area
PARKING_AREA = None
# Plots area where model is working
SHOW_DETECTION_FRAME = 0
# SKIP_FRAMES = 100
FRAME_SIZE = (1080, 1920)


# top-left and bottom-right detection coordinates
X1 = 50
Y1 = 780
X2 = 1500
Y2 = 980

# TL_XY = (X1, Y1)
# BR_XY = (X2, Y2)


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6
