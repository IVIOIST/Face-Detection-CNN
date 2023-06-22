# Face-Detection-CNN
## Overview
- This project is an attempt to create a face detection convolutional neural netowrk.
- This was done by fine tuning a VGG16 model.
- Keep mind that the dataset is me and that it is not guaranteed to work perfectly with you.
- This serves as a learning project and a proof of concept.
- I initially built this with VGG16 but ended up with VGG19 because I had better results with it, I have included both in this repo.
## Prerequisites
- To use this you need to have tensorflow installed and properly configured.
- Have Python 3.9+ installed and added to the system PATH
## Cloning Repository and Installing Dependencies
1. Clone the repository
``` [Terminal]
C:\> git clone https://github.com/IVIOIST/Face-Detection-CNN
```
2. Installing dependencies via requirements.txt
``` [Terminal]
C:\> pip install -r requirements.txt
```
## Breakdown of Model
As I said, this was built on VGG19, the layers are outlined below.
![Alt text](/readmedata/VGG19.png?raw=true "Title")
The classification and bounding box layers are shown here.
![Alt text](/readmedata/layers.png?raw=true "Title")
