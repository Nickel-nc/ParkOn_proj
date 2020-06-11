"""
MRCNN Model Block

MODE:
-- Skips the frames by FRAME_RATE
-- Creates a binary array as output
-- Sends output to Mongo DataBase

Idea (c)  2018 Matterport, Inc.
(c) 2020 @Nickel-nc, @Valdert

Model Finetune:
python model_finetune.py train --dataset=/finetune_dataset --weights=coco
"""

#####################
# Import libs
#####################

import cv2
from mrcnn.model import MaskRCNN
from Settings import *
from mongo_db_module import mongo_init, upload_result
import time
import threading
from threading import Lock
from datetime import datetime

# Buffer for seeking latest frame
def rtsp_cam_buffer(vcap):
    global latest_frame, lo, last_ret
    while True:
        with lo:
            last_ret, latest_frame = vcap.read()

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    return np.array(boxes)

# prints operational info
def op_msg(start, cnt):
    t1 = (datetime.now() - start).seconds
    print(f"App running: {t1} seconds")
    print(f"Processing frame # {cnt}")
    return


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
db = mongo_init()
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)


# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(RTSP_SOURCE)  # Runs only with RTSP_SOURCE
# Sets output with skipped frames

latest_frame = None
last_ret = None
lo = Lock()

t1 = threading.Thread(target=rtsp_cam_buffer,args=(video_capture,),name="rtsp_read_thread")
t1.daemon = True
t1.start()

# Have we sent a free parking space alert yet?
msg_sent = False
# Location of parking spaces
parked_car_boxes = get_car_boxes(INIT_PARKING_ZONES, INIT_IDS)
start_time = datetime.now()
count = 0

# while video_capture.isOpened():

while True :
    if (last_ret is not None) and (latest_frame is not None):
        frame = latest_frame.copy()
        op_msg(start_time, count)

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        # Looks for specific detection area on the screen
        rgb_image = frame[Y1:Y2, X1:X2, ::-1]
        # print(rgb_image.shape)
        # Run the image through the Mask R-CNN model to get results.
        results = model.detect([rgb_image], verbose=0)
        # Mask R-CNN assumes we are running detection on multiple images.
        # We only passed in one image to detect, so only grab the first result.
        r = results[0]
        # We already know where the parking spaces are. Check if any are currently unoccupied.
        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)
        # Assume no spaces are free until we find one that is free
        free_space = False
        space_occupation = np.zeros(len(parked_car_boxes))
        # print("space", space_occupation)
        # print("car_boxes", car_boxes)
        # print("parked car boxes", parked_car_boxes)
        # print("overlaps", overlaps)

        # Loop through each known parking space box
        for i, (parking_area, overlap_areas) in enumerate(zip(parked_car_boxes, overlaps)):
            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
            max_IoU_overlap = np.max(overlap_areas)
            # print("max_IoU_overlap", max_IoU_overlap)
            # Get the top-left and bottom-right coordinates of the parking area
            y1, x1, y2, x2 = parking_area
            y1, x1, y2, x2 = y1 + Y1, x1 + X1, y2 + Y1, x2 + X1
            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than (0.15) using IoU
            if max_IoU_overlap < OVERLAP_THRESH_HOLD:
                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                space_occupation[i] = 1
                # Flag that we have seen at least one open space
                free_space = True
            else:
                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, str(space_occupation), (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

        if WRITE_TO_DB_MODE:
            upload_result(space_occupation, db)
            print("Result: ", space_occupation, "successfully uploaded to db")

        if WRITE_IMG_LOGS:
            fn_gen = f"frame{str(datetime.now()).replace(':','-')}.jpg"
            cv2.imwrite(f"./{STORED_FRAMES_DIR}/{fn_gen}", frame)

        # Hit 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1
        time.sleep(SEEK_RATE)

    else:
        print("unable to read the frame")
        time.sleep(0.5)
        continue

# Clean up everything when finished
video_capture.release()
# video_writer.release()
cv2.destroyAllWindows()