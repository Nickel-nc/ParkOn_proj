"""
MRCNN Model Block

Idea (c)  2018 Matterport, Inc.
(c) 2020 @Nickel-nc, @Valdert

Model Finetune:
python model_finetune.py train --dataset=/finetune_dataset --weights=coco
"""

#####################
# Import libs
#####################

import numpy as np
import cv2
from mrcnn.model import MaskRCNN
from Settings import *
import pandas as pd

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    # car_boxes = []
    # for i, box in enumerate(boxes):
    #     # If the detected object isn't a car / truck, skip it
    #     if class_ids[i] in [3, 8]:
    #         car_boxes.append(box)
    # # Sort boxes list (from left to the right) for x2 coordinate
    # print(car_boxes)
    return np.array(boxes)

# Alternative parking area detection for box geometry from scratch
# Inputs detected cars boxes
def get_parked_boxes(car_boxes, x_pad, car_width, max_load):
    # get blank grid
    parking_boxes = pd.DataFrame(np.zeros([max_load, 4], dtype='int32'))

    y1_last = 0
    x_last = 0
    y2_last = 0

    # fill the correct space of
    for index, row in car_boxes.iterrows():
        free_space_cnt = (row[1] - x_last) // (car_width + x_pad)
        parking_boxes.iloc[free_space_cnt] = row
    return parking_boxes


def detect_parking_area(car_boxes):

    x_pad = 10  # x padding coef based on img perspective
    y_pad = 20  # y padding coef based on img perspective
    X_max = X2 - X1
    car_boxes = pd.DataFrame(car_boxes).sort_values(by=1)  # Sort grids by x1 field

    # get mean car width and height
    car_width = int((car_boxes[3] - car_boxes[1]).mean()) or 100
    car_height = int((car_boxes[2] - car_boxes[0]).mean()) or 100

    # max cars that can be parked in a row
    max_car_load = int((X_max) / (car_width + x_pad * 2))

    y1_last = 0
    x_last = 0
    y2_last = 0

    parking_boxes = get_parked_boxes(car_boxes, x_pad=x_pad, car_width=car_width, max_load=max_car_load)

    for i, row in parking_boxes.iterrows():
        if row[1] - x_last >= car_width or row.any() == 0:
            free_space_cnt = (row[1] - x_last) // car_width  # compute how much cars can enter free space by width
            if free_space_cnt > 1:
                y1 = y1_last + y_pad
                x1 = x_last + x_pad
                y2 = y1 + car_height + y_pad
                x2 = x_last + car_width
                # compute coodinates for new parking box from top right to bottom left (y1 x1 y2 x2)
                parking_boxes.iloc[i] = [y1, x1, y2, x2]
                y1_last = y1
                x_last = x2 + x_pad
                y2_last = y2
            else:  # if only one car can enter free space
                y1 = int((y1_last + row[0]) / 2) + y_pad  # consider y-shifting by perspective ange
                x1 = x_last + x_pad
                y2 = y1 + car_height + y_pad
                x2 = x_last + car_width

                # compute coodinates for new parking box from top right to bottom left (y1 x1 y2 x2)
                parking_boxes.iloc[i] = [y1, x1, y2, x2]

                y1_last = row[0]
                x_last = row[3]
                y2_last = row[2]
        else:
            print("x1 less than avg", row[1] - x_last)
            y1_last = row[0]
            x_last = row[3]
            y2_last = row[2]

    #     boxes = np.append(car_boxes,parking_boxes, axis=0)
    # print(boxes)
    return np.array(parking_boxes)


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None  # INITIAL_PARKING_ZONES

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(RTSP_SOURCE)  # Options: LOCAL_VIDEO_SOURCE, RTSP_SOURCE
# Sets output with skipped frames

"""SKIP FRAME ACTUATE"""
# video_capture.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES)

# Save the detection result video
"""ACTUATE IMAGE SIZE BY CV2 TOOLS"""

# mp4 format writer
# video_writer = cv2.VideoWriter(OUTPUT_SOURCE, -1, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))
# avi format writer
# video_writer = cv2.VideoWriter(OUTPUT_SOURCE, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (int(video_capture.get(3)), int(video_capture.get(4))))

# How many frames of video we've seen in a row with a parking space open
free_space_frames = 0
last_green = False

# Have we sent a free parking space alert yet?
msg_sent = False
frame_count = 0
# Loop over each frame of video
while video_capture.isOpened():
# it = 0
# while it < 1:
    success, frame = video_capture.read()
    if not success:
        break

    frame_count += 1

    # Declare box where detection is working
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[Y1:Y2, X1:X2, ::-1]
    # print(frame.shape)
    # print(rgb_image.shape) # (1080, 1920, 3)

    """CHOOSE THE WINDOW TO LOOK UP FOR BOXES"""

    # rgb_image = rgb_image[:,:500, ::]

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # TEST mode of detection frame mask
    if SHOW_DETECTION_FRAME:
        # Draw a top-left and bottom-right coordinates rectangle (BGR) (x1, y1), (x2, y2)
        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 2)
        # print(f"Detection operating frame: {X1, X2} {Y1, Y2}")

    # TEST MODE of parking area
    if SHOW_PARKING_AREA:
        # Draw a top-left and bottom-right coordinates rectangle (BGR) (x1, y1), (x2, y2)
        cv2.polylines(frame, [PARKING_AREA_PTS], True, (0, 255, 0), 2)

    if parked_car_boxes is None:
        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
        # - r['rois'] are the bounding box of each detected object
        # - r['scores'] are the confidence scores for each detection

        # Auto train car boxes
        if CALC_PARK_AREA:
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            parked_car_boxes = detect_parking_area(car_boxes)
        else:
            parked_car_boxes = get_car_boxes(INIT_PARKING_ZONES, INIT_IDS)
            # parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # print(type(parked_car_boxes))
        # print(parked_car_boxes)
        # parked_car_boxes = PARKING_BOXES
        # Use car boxes from snippet
        # print(parked_car_boxes)
    else:
        # We already know where the parking spaces are. Check if any are currently unoccupied.

        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume no spaces are free until we find one that is free
        free_space = False
        print("parked car boxes", parked_car_boxes)
        # print("overlaps", overlaps)
        space_occupation = np.zeros(len(parked_car_boxes))

        # Loop through each known parking space box
        for i, (parking_area, overlap_areas) in enumerate(zip(parked_car_boxes, overlaps)):
            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
            max_IoU_overlap = np.max(overlap_areas)

            # Get the top-left and bottom-right coordinates of the parking area
            # y1, x1, y2, x2 = parking_area
            y1, x1, y2, x2 = parking_area
            y1, x1, y2, x2 = y1 + Y1, x1 + X1, y2 + Y1, x2 + X1

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU

            if max_IoU_overlap < OVERLAP_THRESH_HOLD:

                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Flag that we have seen at least one open space
                free_space = True

                space_occupation[i] = 1
            else:

                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
        cv2.putText(frame, str(space_occupation), (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.

        # Show the frame of video on the screen

        cv2.imshow('Video', frame)
        # video_writer.write(frame)
        # cv2.imwrite("image.jpg", frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
# video_writer.release()
cv2.destroyAllWindows()
