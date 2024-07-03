## Description

This repository is dedicated to research on the use of polarization imaging
for low-contrast object detection.

To make comparison between RGB and polarization imaging, two different cameras
"Alvium 1800 C-508" and "Mako G-508B POL" were used. By placing them together
inside a homemade box with embedded computing board 2 images from almost
the same angle can be simultaneously got.

To solve detection task YOLOv7 model was used.
More details: https://towardsdatascience.com/yolov7-a-deep-dive-into-the-current-state-of-the-art-for-object-detection-ce3ffedeeaeb

Dataset is available here: https://drive.google.com/drive/folders/1zu769HFwuf9wyzM6uzzYqpJ686-jgwuD?usp=sharing

## Contents

### [`configs`](configs)

Directory that contain train configs for yolov7.

### [`datasets`](datasets)

Package with training datasets modules.

### [`mako_camera`](mako_camera)

Package with utils to work with mako cameras.

### [`utils`](utils)

Package with overall utils.

### [`yolo_scripts`](yolo_scripts)

Directory that contain scripts to work with yolov7. 

### [`yolov7`](yolov7)

Package with yolov7 implementation.
