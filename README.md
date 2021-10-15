# HCAT - Hair Cell Analysis Toolbox

Hcat is a suite of machine learning enabled algorithms for performing common image analyses in the hearing field.
At present, it performs two fully automated analyses: (1) Volumetric hair cell segmentation and (2) 2D hair cell detection. 


This tool is capable of automatically generating cochleograms, determine cell fluorescent intensity statistics, and
investigating tonotopic and morphological trends. 

---
## Quickstart Guide
To install: `pip install hcat`

Segmentation Analysis:
* Run in terminal: `hcat segment "path/to/file.tif"`

Hair Cell Detection Analysis:
* Run in terminal: `hcat detect "path/to/file.tif"`

---

## Requirements
The following requirements are necessary to run `hcat`. 

* Ubuntu 20.04 LTS
* Nvidia GPU with larger than 11 GB of VRAM
* At minimum, 64 GB of RAM
* python 3.8
* pytorch v1.9.0
* torchvision v0.10.0

## Installation
To install hcat, ensure you that **Python Version 3.8** as well as all dependencies properly installed. 
To install the program, run the following in the terminal:

`pip install hcat`

This will install all remaining dependencies as well as prepare the command line interface.

To upgrade `hcat` to the latest version, run the following:

`pip install hcat --upgrade`

---
## Usage
The comand line tool has two entry points: **segment**, and **detect**. 
* **segment** takes in a 3D, multichannel volume of cochlear hair cells and generates unique segmentation masks which which may be used 
* **detect** takes in a 2D, multichannel maximum projection of a cochlea and predicts inner and outer hair cell detection predictions

### Segment

---
`hcat segment` is the entrypoint for volumetric segmentation of hair cells. The program will iteratively segment arbitrarily
sized images, up to every hair cell in a cochlea. This is particularly useful for gene therapy studies, where nuanced measures
of cell intensity may provide insight in the efficacy of a gene therapy construct. To evaluate an image, run the following in the 
command line:

`hcat segment [INPUT] [OPTIONS]`

#### INPUT

The program accepts volumetric confocal images of cochlea in the tiff format. The cells must be stained with a hair cell
specific cytosol stain (usually anti-Myo7a), at either 8-bit or 16-bit resolution. The best performing images are 
one with high signal-to-noise ratio, low background staining, and good separation between hair cells. 
Confocal z-stacks postprocessed with deconvolution algorithms may only aid in segmentation accuracy.  

#### OPTIONS                          
    --channel                         (int) cytosolic channel index, (0 for greyscale)
    --intesnity_reject_threshold      (float) Rejection for objects with mean cytosolic intensity below threshold
    --dtype                           (str) Data type of input image: (uint8 or uint16)
    --unet                            (flag) Generate segmentation masks with U Net + Watershed backbone
    --cellpose                        (flag) Generate segmentation masks with 3D cellpose backbone
    --figure                          (flag) Save a preliminary analysis figure
    --no_post                         (experimental, flag) Disable postprocessing on detection

#### OUTPUT

The program will save two files with the same name and in the same location as the original file: `filename.csv` and
`filename.cochlea`.
* `filename.csv` contains human-readable data on each hair cell segmented in the original image.
* `filename.cochlea` is a dataclass of the analysis which is accessible via the python programing language
  and contains a compressed tensor array of the predicted segmentation mask.

To access `filename.cochela` in a python script:

```python
import torch
cochlea = torch.load('filename.cochlea')

# If the mask is compressed
cochlea.decompress_mask()
predicted_segmentation_mask = cochlea.mask

print('Mask Shape: ', predicted_segmentation_mask.shape)
# Mask Shape: torch.Shape[5000,6000,35]
```

Alternativley you may access the cochlea object via the hcat package:

```python
from hcat.lib.cochlea import Cochlea

cochela = Cochlea.load('filename.cochlea')
```


### Detect 

---
`hcat detect` is the entrypoint for the detection of hair cells from max projection tilescans of a cochlea. 
Hair cell detection is one of the most basic tasks in cochlear image analysis; 
useful for evaluating cochlear trauma, aging, ototoxicity, and noise exposure. To evaluate an image, run the following in
the command line:

`hcat detect [INPUT] [OPTIONS]`

#### INPUT

The program accepts confocal max-projected z-stacks of cochlear hair cells stained with a hair cell specific cytosol stain 
(usually anti-Myo7a) _**and**_  a stereocilia stain (ESPN, phalloidin, etc...). The input image must only have these 2 channels,
_**in this order**_. This may be easiest achieved with the Fiji application. The best performing images will have 
high signal-to-noise ratio and low background staining. 

#### OPTIONS                          
    --cell_detection_threshold        (float) Rejection for objects with mean cytosolic intensity below threshold
    --curve_path                      (str) Path to collection of points for curve estimation
    --dtype                           (str) Data type of input image: (uint8 or uint16)
    --save_fig                        (flag) Render diagnostic figure containing cell detection information
    --save_xml                        (flag) Save detections as xml format compatable with labelImg software
    --pixel_size                      (int) X/Y pixel size in nm
    --cell_diameter                   (int) Rough diameter of hair cell in pixels


#### OUTPUT

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


---

## Common Issues

1. _**The program doesn't predict anything**_: This is most likely a channel issue. The machine learning backbones to each 
model is not only channel specific, but also relies on **specific channel ordering**. Check the `--channel` flag is set
properly for `hcat segment`. For `hcat detect` check that the order of your channels is correct (cytosol then hair bundle).
2. _**The program still doesn't show anything**_: If it is not the channel, then it is likely a datatype issue. Ensure you are
passing in an image of dtype **uint8** or **uint16**. This can be double checked in the `fiji` application by clicking the
`Image` dropdown then clicking `type`, it should show either 8-bit or 16-bit.
3. _**I cannot find the output**_: The program saves the output of each analysis as a CSV file with the same name
in the same location as the original file! Beware, subsequent excecutions of this program will overwrite previous analysis files.