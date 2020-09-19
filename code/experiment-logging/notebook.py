# -*- coding: utf-8 -*-
"""torchvision-experiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FEUcJ4VJQzsehYRL6dxobRfoopsPVlja

# Introduction

In this notebook, we're going to apply experiment tracking to the Object Detection Task on COCO dataset using Torchvision. We are going to use the code from torchvision as is, we won't change anything as the idea is to show how an experiment logging functionality can be brought into some existing training framework.

# Set everything up

In this part, we are going to set up the environment, i.e. download the source
code, initialize functions, download and prepare COCO dataset and its annotations, install and import required libraries, and log in to the Weights & Biases tool.

## Download torchvision code from sources

Using the command below we'll download the detection task code from PyTorch repository. Please, visit [pytorch repository](https://github.com/pytorch/vision) for more information.
"""

!wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/coco_eval.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/coco_utils.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/engine.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/group_by_aspect_ratio.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/train.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/transforms.py && \
 wget https://raw.githubusercontent.com/pytorch/vision/3974cfeb447be9f5ef1fc95e5c320f0498fe68ae/references/detection/utils.py

"""## Download COCO dataset
In this section, we'll download COCO validation images and annotations. We're going to use the validation set for training and validation.

**Please, don't do this in real life as models should be trained on training images and validated on the validation set. Otherwise, you'll face [overfitting](https://https://en.wikipedia.org/wiki/Overfitting)**. 

So we're using validation set for training because the complete dataset is very large and training on it will take a lot of time. While our goal is to show how the experiment logging works and we want to do it in a reasonable amount of time!
"""

!mkdir coco && mkdir coco/annotations

!wget http://images.cocodataset.org/zips/val2017.zip && \
 wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip val2017.zip && unzip annotations_trainval2017.zip

!cp -r val2017 coco/train2017
!cp annotations/instances_val2017.json coco/annotations/instances_val2017.json

!mv val2017 coco/val2017
!mv annotations/instances_val2017.json coco/annotations/instances_train2017.json

"""## Install required libraries"""

!pip install numpy==1.17.3
!pip install tensorboardX==2.0
!pip install wandb==0.8.36

"""## Log in to Weights & Biases

Please go to https://www.wandb.com/ and create an account if you don't have one yet. Then, you need to open https://app.wandb.ai/authorize to get the W&B API key to be able to start streaming logging to W&B. Paste an API key from your profile and hit enter after running the next cell.
"""

!wandb login

"""## Import dependencies"""

import datetime
import math
import os
import os.path as osp
import sys
import time

import tensorboardX
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torchvision.models.detection.mask_rcnn
import wandb

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from engine import _get_iou_types
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from train import get_transform, get_dataset

"""## Initialization

### COCO labels

Let's initialize mapping so as to be able to see human-readable labels.
"""

# COCO labels mapping
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__","person","bicycle","car","motorcycle","airplane","bus",
    "train","truck","boat","trafficlight","firehydrant","N/A","stopsign",
    "parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","N/A","backpack","umbrella","N/A","N/A","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sportsball","kite",
    "baseballbat","baseballglove","skateboard","surfboard","tennisracket",
    "bottle","N/A","wineglass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut",
    "cake","chair","couch","pottedplant","bed","N/A","diningtable","N/A","N/A",
    "toilet","N/A","tv","laptop","mouse","remote","keyboard","cellphone",
    "microwave","oven","toaster","sink","refrigerator","N/A","book","clock",
    "vase","scissors","teddybear","hairdrier","toothbrush",
]

"""### Define main function

Let's take the main function of the code from PyTorch detection as is.
"""

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=5,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

"""# Logging

In this part, we'll add TensorBoard and W&B logging to the original PyTorch detection code.

## Define custom TensorBoard Summary Writer

In the next cell, we'll extend the default Summary writer by adding the function that combines plots to subgroups by adding a prefix to a group of loss dictionary keys.
"""

# create a new class inheriting from tensorboardX.SummaryWriter
class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        super().__init__(log_dir, comment, **kwargs)

    # create a new function that will take dictionary as input and uses built-in add_scalar() function
    # that function combines all plots into one subgroup by a tag
    def add_scalar_dict(self, dictionary, global_step, tag=None):
        for name, val in dictionary.items():
            if tag is not None:
                name = osp.join(tag, name)
            self.add_scalar(name, val.item(), global_step)

"""## Applying logging

### Tracking loss in the training procedure

We want TensorBoard to track the loss. In this section, we'll use our TensorBoard Summary Writer.
"""

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    global global_iter
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        # ### OUR CODE ###
        # let's track the losses here by adding scalars
        logger.add_scalar_dict(
            # passing the dictionary of losses (pairs - loss_key: loss_value)
            loss_dict,
            # passing the global step (number of iterations)
            global_step=global_iter,
            # adding the tag to combine plots in a subgroup
            tag="loss"
        )
        # incrementing the global step (number of iterations)
        global_iter += 1
        # ### END OF OUR CODE ###

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

"""### Visualizing predicted bounding boxes in the validation procedure

In this part, we'll add the code saves the first 50 images with the predicted bounding boxes and labels for each epoch. That way we will be able to see how they are changing during the process.
"""

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # changing these two lines a bit to have iteration number and to keep image tensor
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        img = images[0]

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        # ### OUR CODE ###
        # let's track bounding box and labels predictions for the first 50 images
        # as we hardly want to track all validation images
        # but want to see how the predicted bounding boxes and labels are changing during the process
        if i < 50:
            # let's add tracking images with predicted bounding boxes
            logger.add_image_with_boxes(
                # adding pred_images tag to combine images in one subgroup
                "pred_images/PD-{}".format(i),
                # passing image tensor
                img,
                # passing predicted bounding boxes
                outputs[0]["boxes"].cpu(),
                # mapping & passing predicted labels
                labels=[
                    COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in outputs[0]["labels"].cpu().numpy()
                ],
            )
        # ### END OUR CODE ###
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

"""### Saving config and COCO metrics

Finally, we want to track the metric values and bind them to model hyperparameters.
"""

class CocoEvaluator(CocoEvaluator):
    def summarize(self):
        global config, total_epochs
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            # ### OUR CODE ###
            if iou_type == "bbox":
                # let's add hyperparameters and bind them to metric values
                logger.add_hparams(
                    # passing hyperparameters dictionary (in our case argparse values)
                    config,
                    # passing COCO metrics
                    {
                        "AP/IoU/0.50-0.95/all/100": coco_eval.stats[0],
                        "AP/IoU/0.50/all/100": coco_eval.stats[1],
                        "AP/IoU/0.75/all/100": coco_eval.stats[2],
                        "AP/IoU/0.50-0.95/small/100": coco_eval.stats[3],
                        "AP/IoU/0.50-0.95/medium/100": coco_eval.stats[4],
                        "AP/IoU/0.50-0.95/large/100": coco_eval.stats[5],
                        "AR/IoU/0.50-0.95/all/1": coco_eval.stats[6],
                        "AR/IoU/0.50-0.95/all/10": coco_eval.stats[7],
                        "AR/IoU/0.50-0.95/all/100": coco_eval.stats[8],
                        "AR/IoU/0.50-0.95/small/100": coco_eval.stats[9],
                        "AR/IoU/0.50-0.95/medium/100": coco_eval.stats[10],
                        "AR/IoU/0.50-0.95/large/100": coco_eval.stats[11],
                    },
                    name=".",
                    # passing the current iteration (epoch)
                    global_step=total_epochs,
                )
                # incrementing the number of epochs
                total_epochs += 1
            # ### END OF OUR CODE ###

"""## Experiment setup

Let's create some variables to keep everything organized. Usually, everything is stored in a config file but we use the PyTorch detection code as is and in the project, all hyperparameters are stored using Python argparse. We want to highlight everything that is used for experiment management so in the next cell, we'll create the following variables:



*   config - will be used as config i.e. argument dictionary in this project
*   log_dir - the place where log files will be saved
*   total_epochs - the number of epochs
*   global_iter - the number of iterations
*   logger - custom TensorBoard SummaryWriter that is used for logging
"""

# define variables as globals to have an access everywhere
config = None  # will be replaced to argparse dictionary
total_epochs = 0  # init total number of epochs
global_iter = 0  # init total number of iterations

name = "exp-000"  # init experiment name
log_dir = "experiments"  # init path where tensorboard logs will be stored
# (if log_dir is not specified writer object will automatically generate filename)
# Log files will be saved in 'experiments/exp-000'
# create our custom logger
logger = SummaryWriter(log_dir=osp.join(log_dir, name))

"""## Config initialization

Here, let's use the project's parser with default parameters
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
parser.add_argument('--dataset', default='coco', help='dataset')
parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    help='images per gpu, the total batch size is $NGPU x batch_size')
parser.add_argument('--epochs', default=26, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', default=0.02, type=float,
                    help='initial learning rate, 0.02 is the default value for training '
                          'on 8 gpus and 2 images_per_gpu')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--output-dir', default='.', help='path where to save')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
parser.add_argument(
    "--test-only",
    dest="test_only",
    help="Only test the model",
    action="store_true",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    help="Use pre-trained models from the modelzoo",
    action="store_true",
)

# distributed training parameters
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

"""Here we need to modify the default path with the path where the COCO **validation** dataset was unzipped. 

As the idea is to show how experiment management works in a reasonable time, we'll start with the pre-trained weights and will train the model on the validation set. As was mentioned before, **don't do this in real life as models should be trained on training images and validated on the validation set**. You'll see that metrics are increasing and it is expected because the model is trained and validated on the same data. We're doing this to save time! 

Also, we'll change learning rate to *0.0001* and increase batch size from *2* to *6*.
"""

args = parser.parse_args(["--data-path" ,'/content/coco', '--lr', "1e-4", '--pretrained', '--batch-size', "6"])
config = vars(args)

"""## Weights & Biases synchronization

The last thing that we need to do before training is to connect to the Weights & Biases. Fortunately, we don't need to do much since it can be done by adding just two lines of code:

1. The first line initializes experiment with config and experiment name
2. The second line is used to enable TensorBoard synchronization
"""

wandb.init(config=config, name=name)
wandb.init(sync_tensorboard=True)

"""# Running train/val loop

Here we go! Let's start the ball rolling. Please, visit [Weights & Biases](https://www.wandb.com/) to see how the experiment logging works in real-time!
"""

main(args)