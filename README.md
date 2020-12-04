# ACM-pytorch
[![](https://img.shields.io/badge/building-pass-green.svg?style=flat-square)](https://github.com/Tianfang-Zhang/acm-pytorch)
![](https://img.shields.io/badge/language-Python-blue.svg?style=flat-square)
[![](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)


The unofficial implement of paper "Asymmetric Contextual Modulation for Infrared Small Target Detection" in Pytorch. And this is my [Homepage](https://tianfang-zhang.github.io/).  

# More Official Information: 
[Official code: open-acm](https://github.com/YimianDai/open-acm) and [Homepage of Yimian Dai](https://yimiandai.work/)

# Dependencies
Python = 3.6  
Pytorch = 1.2.0

# Train Command
> python train.py --backbone-mode FPN --fuse-mode BiLocal  
> python train.py --backbone-mode FPN --fuse-mode AsymBi  
> python train.py --backbone-mode FPN --fuse-mode BiGlobal  
> python train.py --backbone-mode UNet --fuse-mode BiLocal  
> python train.py --backbone-mode UNet --fuse-mode AsymBi  
> python train.py --backbone-mode UNet --fuse-mode BiGlobal  

# Experiment Results
Net Mode      | Best IoU | Best nIoU  
---           | ---      | ---
FPN+BiLocal   | 0.6675   | 0.6781
FPN+AsymBi    | 0.6954   | 0.7121
FPN+BiGlobal  | 0.6918   | 0.7010
UNet+BiLocal  | 0.6402   | 0.6800
UNet+AsymBi   | **0.7152**   | **0.7281**
UNet+BiGlobal | 0.6975   | 0.7227




