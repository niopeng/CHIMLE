# Dataloader

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`create_lmdb.py`](/code/data/create_lmdb.py).

## Contents

- `LR_dataset`: only reads LR images in test phase where there is no target images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. It downsamples the HR/LR images to four intermediate resolutions on-the-fly.
- `Colourization_dataset`: reads images with colour and generates gray-scale input images with intermediate resolutions on-the-fly.

## Data Augmentation

We use random crop, random flip/rotation, (random scale) for data augmentation. 
