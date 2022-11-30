# CHIMLE

[Project Page][project] | [Video][video] | [Paper][paper] | [Pre-trained Models](/experiments/pretrained_models)

PyTorch implementation of "CHIMLE: Conditional Hierarchical IMLE for Multimodal Conditional Image Synthesis" (NeurIPS 2022).
CHIMLE is able to:

- (Colourization) Automatic colorizing a grayscale image
- (Super-Resolution) Increase the width and height of images by a factor of 16x
- (Decompression) Recover a plausible image from a heavily compressed image
- (Night-to-Day) Change the scene of the input images from nighttime to daytime

![Alt Text](../website/teaser.gif)
## Dependencies and Installation

- Python 3.7
- [PyTorch 1.4](https://pytorch.org)
- NVIDIA GPU + [CUDA 10.2](https://developer.nvidia.com/cuda-downloads)

To get started, please run the following command:
```sh
$ git clone https://github.com/niopeng/CHIMLE.git
```

[Optional] To set up using pip, please run the following command:
```sh
$ pip install -r requirements.txt
```

[Required For Training] The training code uses [DCI CUDA][dci], to set up:
```sh
$ cd ./code/dciknn_cuda
$ python setup.py install
```

## Training and Testing
Please run the following steps:
1. Prepare datasets. Details can be found [here](/code/data).
2. Change config files located under [options](/code/options).
3. [Optional] Load [pre-trained models](/experiments/pretrained_models).
4. Run training/testing commands:
```sh
// Training
$ python train.py -opt options/train/train_[task].json
// Testing
$ python test.py -opt options/test/test_[task].json
```
Note: Training a CHIMLE model for 16x super-resolution requires 32GB of GPU memory and other tasks with 256x256 target resolution require 24GB of GPU memory. Testing only requires less than 16GB of GPU memory.


## Code Organization
The code consists of the following components:
- `data/`: Dataset/Dataloader definition and useful tools
- `dciknn_cuda/`: Fast k-Nearest Neighbour Search (DCI) interface
- `models/`: Defines Tower Implicit Model (TIM), architecture and [Learned Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity) loss
- `options/`: Training/Testing configurations
- `utils/`: Basic utility functions, logger and progress bar
- `sampler.py`: Hierarchical sampling procedure mentioned in our [paper][paper]
- `train.py`: Main training script
- `test.py`: Main testing script

## Citation
If you find this repository useful for your research, please cite the following work.
```
@inproceedings{peng2022chimle,
   title={CHIMLE: Conditional Hierarchical IMLE for Multimodal Conditional Image Synthesis},
   author={Shichong Peng and Alireza Moazeni and Ke Li},
   booktitle={Advances in Neural Information Processing Systems},
   year={2022}
}
```

[project]:https://niopeng.github.io/CHIMLE/
[paper]: https://arxiv.org/abs/2211.14286
[video]: https://youtu.be/plgPL3XyzRg
[pretrain]: https://github.com/niopeng/CHIMLE/tree/main/experiments/pretrained_models
[options]:https://github.com/niopeng/CHIMLE/tree/main/code/options
[data]:https://github.com/niopeng/CHIMLE/tree/main/code/data
[dci]:https://github.com/niopeng/dciknn_cuda
