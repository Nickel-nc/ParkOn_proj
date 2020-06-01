### Parking Car Object Detection


Features:

- Mask-RCNN Architecture
- Transfer Learn based on coco pretrained weights
- Region Proposal Network (RPN) to scan areas  over the backbone feature map that contain objects

-Mask-RCNN Backbone:

ResNet101 backbone, as a feature extractor. The early layers detect low level features (edges and corners), and later layers successively detect higher level features

Feature Pyramid Network (FPN). FPN improves the standard feature extraction pyramid by adding a second pyramid that takes the high level features from the first pyramid and passes them down to lower layers. By doing so, it allows features at every level to have access to both, lower and higher level features.

- ROI Classifier & Bounding Box Regressor. Runs on the regions of interest (ROIs) proposed by the RPN. And just like the RPN, it generates two outputs for each ROI
 

Mask R-CNN (regional convolutional neural network) is a two stage framework: the first stage scans the image and generates proposals(areas likely to contain an object). And the second stage classifies the proposals and generates bounding boxes and masks.
Passing through the backbone network, the image is converted from 1024x1024px x 3 (RGB) to a feature map of shape 32x32x2048. This feature map becomes the input for the following stages.

Further imporvements:

- Detection Area: The model detection looks for selected region in order to achieve better precise and works faster


model scheme:

<img src="figures/Network.PNG" alt="mrcnn network" width="800"/>

The model generates bounding boxes for each instance of an object in the image:

<img src="output/output.gif" alt="output example" width="800"/>

Model is still training...
