# SCNN lane detection in Pytorch

SCNN is a segmentation-tasked lane detection algorithm, described in ['Spatial As Deep: Spatial CNN for Traffic Scene Understanding'](https://arxiv.org/abs/1712.06080). The [official implementation](<https://github.com/XingangPan/SCNN>) is in lua torch.

This repository contains a re-implementation in Pytorch.

<br/>


## Demo Test

For single image demo test:

```shell
python demo_test.py   -i demo/demo.jpg 
                      -w experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth 
                      [--visualize / -v]
```

![](demo/demo_result.jpg "demo_result")



<br/>

## Train 

1. Specify an experiment directory, e.g. `experiments/exp0`. 

2. Modify the hyperparameters in `experiments/exp0/cfg.json`.

3. Start training:

   ```shell
   python train.py --exp_dir ./experiments/exp0 [--resume/-r]
   ```

4. Monitor on tensorboard:

   ```bash
   tensorboard --logdir='experiments/exp0'
   ```

**Note**


- My model is trained with `torch.nn.DataParallel`. Modify it according to your hardware configuration.
- Currently the backbone is vgg16 from torchvision. Several modifications are done to the torchvision model according to paper, i.e., i). dilation of last three conv layer is changed to 2, ii). last two maxpooling layer is removed.



<br/>

## Evaluation

* CULane Evaluation code is ported from [official implementation](<https://github.com/XingangPan/SCNN>) and an extra `CMakeLists.txt` is provided. 

  1. Please build the CPP code first.  
  2. Then modify `root` as absolute project path in `utils/lane_evaluation/CULane/Run.sh`.

  ```bash
  cd utils/lane_evaluation/CULane
  mkdir build && cd build
  cmake ..
  make
  ```

  Just run the evaluation script. Result will be saved into corresponding `exp_dir` directory, 

  ``` shell
  python test_CULane.py --exp_dir ./experiments/exp10
  ```

  

* Tusimple Evaluation code is ported from [tusimple repo](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py).

  ```Shell
  python test_tusimple.py --exp_dir ./experiments/exp0
  ```


## Acknowledgement

This repos is build based on [official implementation](<https://github.com/XingangPan/SCNN>).

