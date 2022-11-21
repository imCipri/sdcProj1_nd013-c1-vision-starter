# Object Detection in an Urban Environment
In this project, a CNN is created to detect and classify objects from the Waymo dataset.
The dataset is constituted of images of urban environments, where three different labels are assigned: vehicles (1), pedestrians (2), and cyclists (4).

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/
    - train: contain the train data (80 files)
    - val: contain the val data (10 files)
    - test - contains 10 files to test the model and create inference videos
```

### Experiments
The experiments folder is organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment1/ - used random_adjust brightness and random_crop_image to augment the data
    - experiment2/ - tested the rmsprop optimizer and restored original random_crop_image
    - experiment3/ - added the random_adjust_contrast to augment the data
    - label_map.pbtxt
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/` directory to explore the dataset! This is the most important task of any machine learning project. 
To do so, open the `Exploratory Data Analysis` notebook. 
In this notebook, the `display_instances` function display batches of 10 images and annotations using `matplotlib`.

Results:
- The most common factor in the major part of the images is the presence of occlusions. This can be present in the following forms:
	a. weather (rain, fog)
	b. light reflexes
	c. other objects. It is indeed common to see traffic scenes, with a lot of cars overimposed, and a lot of pedestrians going together.
- The photos are taken both during day and night. In some cases, we can find very dark scenes, where the only source of light comes from the car.
- In proportion, there are a lot of cars with respect to the pedestrians and the cyclists. The cyclists are very rare.
- There are images without objects.
- Some images present labels for objects that are very small due to the distance from the camera.

Following this analysis, it is possible to make the following assumptions regarding data augmentations:
- The use of patches to increase occlusion would not be beneficial due to the presence of small or almost-hidden objects. Moreover, there are a lot of cases with occlusions.
- The only useful augmentation I can think of is the adjustment of the brightness to cover the presence of very dark images.


### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/` on a new terminal.

Results:
The following plots are taken from tensorboard:
![Tensorboard_screenshots img](/tensorboard_screenshots/reference_loss.png "Reference loss")
![Tensorboard_screenshots img](/tensorboard_screenshots/reference_performance.png "Reference performance")

The images shows that, even if the classification loss is low, the total loss is very high. This is reflected on the precision that is equal to 0. The model finds difficult to localize the majority of the objects. Moreover, the validation losses are higher than the training ones. Therefore, there seems to be a little of overfitting to the training data.

### Improve the performances

The initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

Results:

EXPERIMENT 1
In this first experiment to improve the data, two augmentations are added: random_adjust_brightness and random_crop_image (modified from the already existing one).
The model performs slightly worse than the reference one. The images can be found in tensorboard_screenshots folder. Since the augmentations did not helped for improving the performances, in the next experiment we tried a different optimizer.


EXPERIMENT 2
In this experiment, the optimizer is changed to try to improve the overall performance. By using the RMSProp Optimizer (https://github.com/tensorflow/models/blob/master/research/object_detection/protos/optimizer.proto), we were able to improve the performance, as shown in the following screenshots:

![Tensorboard_screenshots img](/tensorboard_screenshots/exp2_loss.png "Exp2 loss")
![Tensorboard_screenshots img](/tensorboard_screenshots/exp2_performance.png "Exp2 performance")

Next is an example of the predictions of the model:
![Tensorboard_screenshots img](/tensorboard_screenshots/exp2_esPred.png "Exp2 prediction example")

There is still a lot of improvement to be made.
What we can see from the screenshots is that the model performs better than before. Moreover, the validation losses are still higher than the training ones, but the difference is equal or lower than the reference case. Therefore, the overfitting has been reduced. The main reason behind the low performance can be mainly due to the reduced dataset (only 80 images for the training) that can cause the overfitting.

EXPERIMENT 3
Finally in this experiment, we tried to improve more by adding the random_adjust_contrast as augmentation. We thought it could be useful to increase the borders of the single objects, hopefully helping the network detecting features.
However, the performance is worse than exp 2. Moreover, we ran it for 3.5k steps to see if more time could be beneficial.

FINAL CONSIDERATIONS
The experiments have been carried out for just 2.5k steps. The plots shows that the plateaux has not been reached yet. This limitation is mainly due to limited resources on my local desktop. Moreover, a quicker test resulted in more tests. The ones reported in this readme and loaded on the repository are the most significant ones.

### Creating an animation
#### Export the trained model
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```
This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

To conclude, this is a video of the experiment 2 inferences:
![Tensorboard_screenshots video](/tensorboard_screenshots/animation_exp2.gif "Exp2 inference video")
