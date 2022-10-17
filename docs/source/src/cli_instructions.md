# Use the Comand Line Interface

Hcat has one CLI entrypoint:
* **detect** takes in a 2D, multichannel maximum projection of a cochlea and predicts inner and outer hair cell detection predictions

## Detect

`hcat detect` is the entrypoint for the detection of hair cells from max projection tilescans of a cochlea.
Hair cell detection is one of the most basic tasks in cochlear image analysis;
useful for evaluating cochlear trauma, aging, ototoxicity, and noise exposure. To evaluate an image, run the following in
the command line:

`hcat detect [INPUT] [OPTIONS]`

### INPUT

The program accepts confocal max-projected z-stacks of cochlear hair cells stained with a hair cell specific cytosol stain
(usually anti-Myo7a) _**and**_  a stereocilia stain (ESPN, phalloidin, etc...). The input image must only have these 2 channels. This may be easiest achieved with the Fiji application. The best performing images will have
high signal-to-noise ratio and low background staining.

### OPTIONS
    --curve_path                      (str) Path to collection of points for curve estimation
    --cell_detection_threshold        (float) Rejection for objects with mean cytosolic intensity below threshold
    --nms_threshold                   (float) Threshold [0, ..., 1] of allowable bounding box overlap
    --dtype                           (str) Data type of input image: (uint8 or uint16)
    --save_xml                        (flag) Save detections as xml format compatable with labelImg software
    --save_fig                        (flag) Render diagnostic figure containing cell detection information
    --save_png                        (flag) Saves a png image of analzed image
    --pixel_size                      (int) X/Y pixel size in nm
    --cell_diameter                   (int) Rough diameter of hair cell in pixels
    --predict_curvature               (int) Enables whole cochlea curvature estimation and cell frequency assignment
    --silent                          (float) Suppresses most of HCAT's logging

### OUTPUT

The program will save two files with the same name and in the same location as the original file: `filename.csv` and
`filename.cochlea`.
* `filename.csv` contains human-readable data on each hair cell segmented in the original image.
* `filename.cochlea` is a dataclass of the analysis which is accessible via the python programing language
  and contains a compressed tensor array of the predicted segmentation mask.

To access `filename.cochela` in a python script:

```python
import torch
from hcat.lib.cell import Cell
from typing import List

# Detected cells are stored as "Cell" objects 
cochlea = torch.load('filename.cochlea')
cells: List[Cell] = cochlea.cells

# To access each cell:
for cell in cells:
    print(cell.loc, cell.frequency) #location (x, y, z); frequency (Hz)
```

## Common Issues

1. _**The program doesn't predict anything**_: This is most likely a channel issue. The machine learning backbones to each
   model is not only channel specific, but also relies on **specific channel ordering**. Check the `--channel` flag is set
   properly for `hcat segment`. For `hcat detect` check that the order of your channels is correct (cytosol then hair bundle).
2. _**The program still doesn't show anything**_: If it is not the channel, then it is likely a datatype issue. Ensure you are
   passing in an image of dtype **uint8** or **uint16**. This can be double checked in the `fiji` application by clicking the
   `Image` dropdown then clicking `type`, it should show either 8-bit or 16-bit.
3. _**I cannot find the output**_: The program saves the output of each analysis as a CSV file with the same name
   in the same location as the original file! Beware, subsequent excecutions of this program will overwrite previous analysis files.
