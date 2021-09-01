# Searchable-CBAM
This work is based on:
ZAM: Zero parameter Attention Module (https://github.com/developer0hye/ZAM)
It is ispired from [BAM](https://arxiv.org/abs/1807.06514) and [CBAM](https://arxiv.org/pdf/1807.06521.pdf).

The work is testing whether it is possible to select a proper combination of Channel Attention module and Spatial Attention module for network.

## Searchable CBAM Module
We consider 4 possible combination of the attention module:
CA+SA, SA+CA, CA, SA, (CA,SA), skip connection
Here (,) is the parallel operation and + is the sequential operation.


## Experimental Results

### Dataset: CIFAR- 100

This dataset is just like the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

Please see the report "Attention_Module.pdf" for the details of the testing results.


## Run

```
python train_search_structure.py -net resnetcbamss18_2
python train_search_structure.py -net resnetmvamss18_2
python train_search_structure.py -net resnetssam18

``` 

## Reference

- Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
- Paper: [BAM: Bottleneck Attention Module](https://arxiv.org/abs/1807.06514)
- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- Repository: [Jongchan/attention-module](https://github.com/Jongchan/attention-module)
- Repository: [luuuyi/CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)
- Repository: [kobiso/CBAM-tensorflow](https://github.com/kobiso/CBAM-tensorflow)
- Repository: [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- Repository: [marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet)
- Dataset: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

