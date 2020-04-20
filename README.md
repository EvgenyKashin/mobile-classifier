# Mobile classifier
## TLDR
- CelebA dataset
- class weighted datasampler for imbalance labels
- many small models have been tested
- a bit of augmentation, AdamW, exponential lr decay, fp16, imagenet pretraining

## Quick start
**Training:**
```
sudo chmod +x train.sh test.sh predict.sh
sudo ./train.sh /mnt/datasets/celeba shufflenet_v2_x1_0
```
**Testing:**
```
sudo ./test.sh /mnt/datasets/celeba lightning_logs/shufflenet_v2_x1_0/version_0/checkpoints/epoch\=1.ckpt
```
**Predicting on folder:**
```
sudo ./predict.sh example_data_glasses/without_glasses
```
## Dataset
CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html Download
 `img_align_celeba` and `list_attr_celeba.txt` and save it, for example, at
 `/mnt/datasets/celeba`

## Environment
Install docker 19.03 or newer

### Build or pull docker image
`sudo docker pull digitman/mobile_classifier` or
`sudo docker build -t mobile_classifier . `

### Run contaner
```sudo docker run --rm -it --gpus '"device=1"' -v `pwd`:/workspace \
    -v /mnt/datasets/celeba:/data -p 8089:8089 \
    --shm-size=2gb mobile_classifier /bin/bash```

### For EDA
**Run inside docker:**

`jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 8089`

### Training models
```sudo docker run --rm -it --gpus '"device=1"' -v `pwd`:/workspace \
    -v /mnt/datasets/celeba:/data \
    --shm-size=2gb mobile_classifier python train.py```
    
or change to:
`python train.py --model_architecture mnasnet0_5`

Tensorboard logs:

`pip install tensorboad`

`tensorboard --logdir lightning_logs/ --host 0.0.0.0 --port 8097`

### Testing models and saving them in fp16
**Run inside docker:**

`python test.py --checkpoint_path lightning_logs/squeezenet1_1/version_2/checkpoints/epoch\=2.ckpt`

### Predict on folder
**Run inside docker:**

Only the CPU is used for prediction.

Check on CelebA: `python predict.py --folder_path /data/img_align_celeba --predict_max_count 100`
or

`python predict.py --folder_path example_data_glasses/without_glasses`
or

`python predict.py --folder_path example_data_glasses/with_glasses`

## Models comparison
My model of choice is **shufflenet_v2_x1_0** - quite small, not very fast, but
 very accurate. But task is easy and even squeezenet1_1 will be enough.

|Model name|Weighs(fp16)|forward(cpu)|accuracy|f1|roc auc|
|---|---|---|---|---|---|
|squeezenet1_1|**1.4Mb**|9ms|0.990|0.977|0.986|
|mnasnet0_5|1.9Mb|9ms|0.985|0.965|0.970|
|mobilenetv3_small_minimal_100|2.1Mb|**6ms**|0.988|0.972|0.984|
|**shufflenet_v2_x1_0**|2.5Mb|12ms|0.993|**0.984**|**0.990**|
|mobilenetv3_small_100|3Mb|10ms|0.990|0.978|0.987|
|mobilenet_v2|4.4Mb|14ms|0.991|0.980|0.990|
|efficientnet_lite0|6.6Mb|16ms|**0.993**|0.983|0.989|

Forward pass speed was tested on i7-6800K CPU, mean of 100 separate calls is
 presented.

## Pretrained models

## Notebooks

## FP16
Nvidia apex was used for fp16 training. It was tested, that accuracy of
 training won't get worse. Also the final model weights are stored in fp16
 for x2  less disk space. It is also tested, that after loading the model 
 the accuracy of the model is the same. Model prediction  in standard fp32 mode.
   
It is also possible to use fp16 during inference with apex, but you need a GPU.
Another option for fp16 inference is onnx - see next.

## More optimizations
I also tried onnx conversion with fp16. The inference speed doesn't increase
  significantly(~10%). But it can be used for mobile devices and
 may give an increase there. For fp16 (and may be int8 in the future)
 inference I tried fuse conv2d layers with batch-norm layers(mnasnet0_5 and
 shufflenet_v2_x1_0 it is different for each model, because of it I tried
 just for that 2). But it doesn't work with default onnx export function and
 more work needs to be done there. Without batch norm fusing it may be
 hard to achieve high accuracy for some models with batch-norm in fp16 and int8
 inference. But for shufflenet_v2_x1_0 everything is fine.

## Future work
In general, the accuracy is very good, because the task is very simple. More
 important next steps are further compressing the model and increasing speed.
 
- Try pruning(`torch.nn.utils.prune as prune`). I tried it before in
 image2image network, and it can give a more sparse network(~30% zeros), which
 can help compress (gzip) network weights and save disk space. Probably
 for image classification it can be pruned even further.
- Try int8 quantization(`torch.quantization`). I have already added fusing conv
 and batch-norm. This can decrease disk space as well as improve model speed.
- Use multitask learning. Now only one label is used('Eyeglasses'), but
 CelebA has many labels and learning network on all of them can improve
 metrics in the target task.
- My experiments show, that imagenet pretraining was very important for this
 task(data is similar to imagenet). It can be possible to use even smaller
 networks(may be x10 smaller), but you need to pretrain it firstly on ImageNet.
- Add experiments with smaller image input sizes (<160)
- Add experiments with cropping only eyes part of the image by keypoints and
 training only on this data.
- Make model more effective on a certain end device(could we use ANE on iPhone
 and so on) by tuning architecture.
- Add more clever augmentations.
- Now model is trained on already cropped data. But it might be better to use
 "your" face detector to match image distribution during training and during
 using the model in production.
- As experiments show, there are errors in dataset labels. A clean dataset can
 be the easiest way to further improve quality.
- Now accuracy, f1 score and roc auc were the main metrics for choosing the best
 model on the test set. But it could be possible to use directly precison and
  recall to make fine grained choice of the threshold value of the classifier.
### TODO:
- make readme 'models' table generation and all model benchmarking automatically
- replace 'print' to 'logger.log'
