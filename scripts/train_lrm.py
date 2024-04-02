'''
    Adapted from https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/train_ssl.py
    
    Further adapted from https://github.com/harvard-visionlab/model-rearing-workshop/model_rearing_workshop/train_ssl.py

    See that file for more attribution details. We stand on the shoulders of giants. 

'''
import sys
import torch
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import time
import json
import uuid
import ffcv
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
import wandb

from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields import IntField, RGBImageField

# we assume model_rearing_workshop has been pip installed, e.g. from the top directory of model-rearing-workshop: pip install -e .
from model_rearing_workshop.utils.utils import LARS, cosine_scheduler, learning_schedule
from model_rearing_workshop.utils.remote_storage import RemoteStorage

from model_rearing_workshop.models import LinearProbes, MLP, task_networks_lrm

from numba.core.config import NUMBA_NUM_THREADS

# we assume lrm_models has been pip installed, e.g. from the top directory: pip install -e .
from lrm_models.utils import get_random_name
from lrm_models.config import WANDB_ENTITY, WANDB_PROJECT, SAVE_DIR

from pdb import set_trace

import inspect

Section('model', 'model details').params(
    alpha=Param(float, 'weighting across time steps for LRMs (t1: (1-alpha), t2: alpha), between 0 and 1', default=0.5),
    forward_passes=Param(int, 'number of forward passes. forward_passes>2 only comp. with alpha=0.5', default=2),
    mlp=Param(str, 'mlp', default='8192-8192-8192'),
    lrm_ind=Param(int, 'which LRM model to use', default=3),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=64),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=30),
    start_ramp=Param(int, 'when to start interpolating resolution', default=10)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', default=""),
    num_classes=Param(int, 'The number of image classes', default=1000),
    num_workers=Param(int, 'The number of workers', default=NUMBA_NUM_THREADS - 2),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
)

Section('vicreg', 'Vicreg').enable_if(lambda cfg: cfg['training.loss'] == 'vicreg').params(
    sim_coeff=Param(float, 'VicREG MSE coefficient', default=25),
    std_coeff=Param(float, 'VicREG STD coefficient', default=25),
    cov_coeff=Param(float, 'VicREG COV coefficient', default=1),
)

Section('simclr', 'simclr').enable_if(lambda cfg: cfg['training.loss'] == 'simclr').params(
    temperature=Param(float, 'SimCLR temperature', default=0.5),
)

Section('barlow', 'barlow').enable_if(lambda cfg: cfg['training.loss'] == 'barlow').params(
    lambd=Param(float, 'Barlow Twins Lambd parameters', default=0.0051),
)

Section('byol', 'byol').enable_if(lambda cfg: cfg['training.loss'] == 'byol').params(
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('logging', 'how to log stuff').params(
    incubator=Param(str, 'literally just the name of this file', default='train_ssl.py'),
    base_fn=Param(str, 'base filename', default=get_random_name()),
    folder=Param(str, 'log location; if not specified, we use a random string (the base_fn)', default=''),
    bucket_name=Param(str, 's3 bucket storage location', default=''),
    bucket_subfolder=Param(str, 's3 subfolder for storing logs and weights', default=''),
    log_level=Param(int, '0 if only at end 1 otherwise', default=2),
    checkpoint_freq=Param(int, 'When saving checkpoints', default=5),
    use_wandb=Param(int, 'use wandb?', default=0),
)

Section('logging.wandb', 'wandb options').params(
    project=Param(str, 'wandb project name', default=WANDB_PROJECT),
    entity=Param(str, 'wandb entity', default=WANDB_ENTITY),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    eval_freq=Param(float, 'number of epochs', default=1),
    batch_size=Param(int, 'The batch size', default=512),
    num_crops=Param(int, 'number of crops?', default=1),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'lars'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=100),
    base_lr=Param(float, 'base learning rate', default=0.0005),
    end_lr_ratio=Param(float, 'ratio of ending LR to starting LR', default=0.001),
    label_smoothing=Param(float, 'label smoothing parameter', default=.1),
    distributed=Param(int, 'is distributed?', default=1),
    clip_grad=Param(float, 'gradient clip norm value', default=0),
    loss=Param(str, 'which loss function to use', default="simclr"),
    train_probes_only=Param(int, 'load linear probes?', default=0),
    stop_early_epoch=Param(int, 'For debugging, stop afer this many epochs (ignored if less than 1)', default=0),
    use_amp=Param(int, 'use automatic mixed precision?', default=1),
    subset=Param(float, 'subset of data to use for faster prototyping', default=None),
    eps=Param(float, 'epsilon for AdamW. higher = less likelihood of nans', default=1e-4),
)

Section('dist', 'distributed training options').params(
    use_submitit=Param(int, 'enable submitit', default=0),
    world_size=Param(int, 'number gpus', default=1),
    ngpus=Param(int, 'number of gpus per node', default=4),
    nodes=Param(int, 'number of nodes', default=1),
    comment=Param(str, 'comment for slurm', default=''),
    timeout=Param(int, 'timeout', default=2800),
    partition=Param(str, 'partition', default="kempner"),
    account=Param(str, 'account', default="kempner_Alvarez_Lab"),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='58492')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

################################
##### Some Miscs functions #####
################################

def get_model(model, distributed):
    if distributed:
        return model.module
    else:
        return model

def has_argument(func, arg_name):
    """
    Check if a function or method has a specific argument.
    
    :param func: The function or method to inspect.
    :param arg_name: Name of the argument to check for.
    :return: Boolean indicating whether the argument is in the function's signature.
    """
    signature = inspect.signature(func)
    return arg_name in signature.parameters

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    
    # Get the full path to the current script.
    current_script_path = Path(__file__).resolve()

    # Get the directory containing the current script.
    current_script_directory = current_script_path.parent

    # Construct the path to the 'checkpoint' directory in the same location as the current script.
    checkpoint_path = current_script_directory / "checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)
    print("checkpoint_path: ", checkpoint_path, Path(checkpoint_path).is_dir())
    
    if Path(checkpoint_path).is_dir():
        p = Path(f"{checkpoint_path}/{user}/experiments")
        p.mkdir(exist_ok=True, parents=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    x_list = GatherLayer.apply(x.contiguous())
    return ch.cat(x_list, dim=0)

class GatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

################################
##### Loss definitions #####
################################

class SimCLRLoss(nn.Module):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.0, 
    weight decay of 1e-6 and a temperature of 0.15.
    """
    @param('simclr.temperature')
    def __init__(self, batch_size, world_size, gpu, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size).to(gpu)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = ch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.world_size

        if self.world_size > 1:
            z_i = ch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = ch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = ch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = ch.matmul(features, features.T)/ self.temperature

        sim_i_j = ch.diag(sim, batch_size * self.world_size)
        sim_j_i = ch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = ch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        logits = ch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = ch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (- logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim, num_entropy

class VicRegLoss(nn.Module):
    """
    ViCREG Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    """
    @param('vicreg.sim_coeff')
    @param('vicreg.std_coeff')
    @param('vicreg.cov_coeff')
    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_i, z_j, return_only_loss=True):
        # Repr Loss
        repr_loss = self.sim_coeff * F.mse_loss(z_i, z_j)
        std_loss = 0.
        cov_loss = 0.

        # Std Loss z_i
        x = gather_center(z_i)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_i
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_i.size(1))
        
        # Std Loss z_j
        x = gather_center(z_j)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_j
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_j.size(1))

        std_loss = std_loss / 2.

        loss = std_loss + cov_loss + repr_loss
        if return_only_loss:
            return loss
        else:
            return loss, repr_loss, std_loss, cov_loss

class BarlowTwinsLoss(nn.Module):
    @param('barlow.lambd')
    def __init__(self, bn, batch_size, world_size, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
        ch.distributed.all_reduce(c)

        on_diag = ch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class ByolLoss(nn.Module):
    @param('byol.momentum_teacher')
    def __init__(self, momentum_teacher):
        super().__init__()
        self.momentum_teacher = momentum_teacher

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.chunk(2)
        teacher_out = teacher_output.detach().chunk(2)

        student_out_1, student_out_2 = student_out
        student_out_1 = F.normalize(student_out_1, dim=-1, p=2)
        student_out_2 = F.normalize(student_out_2, dim=-1, p=2)
        teacher_out_1, teacher_out_2 = teacher_out
        teacher_out_1 = F.normalize(teacher_out_1, dim=-1, p=2)
        teacher_out_2 = F.normalize(teacher_out_2, dim=-1, p=2)
        loss_1 = 2 - 2 * (student_out_1 * teacher_out_2.detach()).sum(dim=-1)
        loss_2 = 2 - 2 * (student_out_2 * teacher_out_1.detach()).sum(dim=-1)
        return (loss_1 + loss_2).mean()
    
def check_scaler_scale(scaler, scale=128):
    # try to solve NAN issue
    if hasattr(scaler, '_scale') and scaler._scale < 128:
        scaler._scale = ch.tensor(scale).to(scaler._scale)

################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.batch_size')
    @param('training.label_smoothing')
    @param('training.loss')
    @param('training.train_probes_only')
    @param('training.epochs')
    @param('data.train_dataset')
    @param('data.val_dataset')
    @param('data.num_classes')
    def __init__(self, gpu, ngpus_per_node, world_size, dist_url, distributed, batch_size, label_smoothing, loss, train_probes_only, 
                 epochs, train_dataset, val_dataset, num_classes):
        
        self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.seed = 50 + self.rank
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()
        self.start_epoch = 0
        
        # Create DataLoader
        self.train_dataset = train_dataset
        self.index_labels = 1
        self.train_loader = self.create_train_loader_ssl(train_dataset)
        self.num_train_exemples = self.train_loader.indices.shape[0]
        self.num_classes = num_classes
        self.val_loader = self.create_val_loader(val_dataset)
        self.vis_loader = self.create_val_loader(val_dataset, subset=0.1)
        print("NUM TRAINING EXAMPLES:", self.num_train_exemples)
        
        # Create SSL model, scaler, and optimizer
        model_config = {key.replace('model.', '') : value for key, value in self.params_dict().items() if key.startswith('model.')}
        model_config.pop('alpha')
        self.model, self.scaler = self.create_model_and_scaler(model_config=model_config)
        print(self.model)
        self.num_features = get_model(self.model, distributed).num_features        
        self.n_probe_layers = len(get_model(self.model, distributed).mlp_spec.split("-"))
        print("NUM PROBE LAYERS:", self.n_probe_layers)
        
        self.initialize_logger()
        self.initialize_remote_logger()
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer()
        
        # Create linear probes (trained without label smoothing)
        self.sup_loss = nn.CrossEntropyLoss()
        self.probes = LinearProbes(get_model(self.model, distributed).mlp_spec, num_classes=self.num_classes)
        self.probes = self.probes.to(memory_format=ch.channels_last)
        self.probes = self.probes.to(self.gpu)
        if distributed:
            self.probes = ch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu])
        self.optimizer_probes = ch.optim.AdamW(self.probes.parameters(), lr=1e-4)
        
        # Load models if checkpoint exists
        self.load_checkpoint()
                
        # Define SSL loss
        self.do_network_training = False if train_probes_only else True
        self.teacher_student = False
        self.supervised_loss = False
        self.loss_name = loss
        if loss == "simclr":
            self.ssl_loss = SimCLRLoss(batch_size, world_size, self.gpu).to(self.gpu)
        elif loss == "vicreg":
            self.ssl_loss = VicRegLoss()
        elif loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(get_model(self.model, distributed).bn, batch_size, world_size)
        elif loss == "byol":
            self.ssl_loss = ByolLoss()
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler(model_config=model_config)
            print("TeacherNetwork:")
            print(self.teacher)
            msg = self.teacher.module.load_state_dict(get_model(self.model, distributed).state_dict())
            print(msg)
            self.momentum_schedule = cosine_scheduler(self.ssl_loss.momentum_teacher, 1, epochs, len(self.train_loader))
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif loss == "ipcl":
            print("Loss not available, YET")
            exit(1)
        elif loss == "supervised":
            self.supervised_loss = True     
            self.add_supervised_meters()
        else:
            print("Loss not available")
            exit(1)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('training.eps')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, eps):
        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4, eps=eps)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        self.optim_name = optimizer

    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    @param('training.loss')
    @param('training.use_amp')
    @param('training.subset')
    def create_train_loader_ssl(self, train_dataset, num_workers, batch_size,
                                distributed, in_memory, loss, use_amp, 
                                subset=None,
                                ):
        if distributed:
            this_device = f'cuda:{self.gpu}'
        else:
            this_device = 'cuda'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if use_amp else np.float32),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if use_amp else np.float32),
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        if loss == 'supervised':
            pipelines={
                'image': image_pipeline_big,
                'label': label_pipeline,
            }
            custom_fields = {
                            'image': RGBImageField,
                            'label': IntField,
                        }
            custom_field_mapper=None
        else:
            pipelines={
                'image': image_pipeline_big,
                'label': label_pipeline,
                'image_0': image_pipeline_big2
            }
            custom_fields = {
                            'image': RGBImageField,
                            'image_0': RGBImageField,
                            'label': IntField,
                        }
            custom_field_mapper={"image_0": "image"}

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_fields=custom_fields,
                        custom_field_mapper=custom_field_mapper)
        
        if subset is not None:
            len_dset = loader.indices.shape[0]
            indices = np.random.random_integers(0, len_dset, int(subset * len_dset))
            loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        indices=indices,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_fields=custom_fields,
                        custom_field_mapper=custom_field_mapper,
                        )



        return loader

    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    @param('training.use_amp')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed, use_amp, subset=None):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16 if use_amp else np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        custom_fields={
                            'image': RGBImageField,
                            'label': IntField,
                        },
                        distributed=distributed)
        
        if subset is not None:
            len_dset = loader.indices.shape[0]
            indices = np.random.random_integers(0, len_dset, int(subset * len_dset))
            loader = Loader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            order=order,
                            drop_last=False,
                            indices=indices,
                            pipelines={
                                'image': image_pipeline,
                                'label': label_pipeline
                            },
                            custom_fields={
                                'image': RGBImageField,
                                'label': IntField,
                            },
                            distributed=distributed)

        return loader

    @param('training.epochs')
    @param('training.stop_early_epoch')
    @param('logging.log_level')
    def train(self, epochs, stop_early_epoch, log_level):
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_exemples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            res = self.get_resolution(epoch)
            self.res = res
            self.decoder.output_size = (res, res)
            self.decoder2.output_size = (res, res)
            train_loss, stats = self.train_loop(epoch)
            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch,
                }
                self.log(dict(stats, **extra_dict, phase='train'), 'train')
            self.eval_and_log(stats, dict(**extra_dict, phase='val'))
            # Run checkpointing
            self.checkpoint(epoch + 1)
            
            # debugging
            if stop_early_epoch>0 and (epoch+1)>=stop_early_epoch: break
            
        if self.rank == 0:
            self.save_checkpoint(epoch + 1)
            ch.save(dict(
                epoch=epoch,
                state_dict=self.model.state_dict(),
                probes=self.probes.state_dict(),
                params=self.params_dict()
            ), self.log_folder / 'final_weights.pth')
            
            if self.remote_store is not None:
                self.remote_store.upload_final_results()
            
    def params_dict(self):
        params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
        }
        return params
    
    def eval_and_log(self, stats, extra_dict={}):
        stats = self.val_loop()
        if stats is not None:
            self.log(dict(stats, **extra_dict), 'val')
        return stats
    
    @param('model.alpha')
    @param('model.forward_passes')
    def get_loss_weighting(self, step, forward_passes, alpha):
        """
        step: current step (1-based)
        forward_passes: total number of forward_passes
        alpha: weighting
        """
        if alpha == 0.5:
            return 0.5
        else:
            assert forward_passes == 2 and step in [1,2], "Only 2 forward_passes are supported with alpha != 0.5"
            if step == 1:
                return alpha
            else:
                return 1-alpha

    
    @param('training.loss')
    @param('training.use_amp')
    @param('training.distributed')
    def create_model_and_scaler(self, loss, use_amp, distributed, model_config):
        scaler = GradScaler(enabled=bool(use_amp), growth_interval=100)
        model = task_networks_lrm.__dict__["alexnet2023_nobn_lrm"](loss=loss, **model_config)
        assert model.loss == loss, f"TaskNetwork loss ({model.loss}) must match training.loss ({loss})"
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    @param('training.train_probes_only')
    def load_checkpoint(self, train_probes_only):
        if (self.log_folder / "model.pth").is_file():
            if self.rank == 0:
                print("resuming from checkpoint")
            ckpt = ch.load(self.log_folder / "model.pth", map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if not train_probes_only: # train_probes_only means train_probes_only_from_scratch
                self.probes.load_state_dict(ckpt["probes"])
                self.optimizer_probes.load_state_dict(ckpt["optimizer_probes"])
            else:
                # training probes only from checkpoint
                self.start_epoch = 0

    @param('logging.checkpoint_freq')
    def checkpoint(self, epoch, checkpoint_freq):
        if self.rank != 0 or epoch % checkpoint_freq != 0:
            return
        self.save_checkpoint(epoch)
    
    @param('training.train_probes_only')
    def save_checkpoint(self, epoch, train_probes_only):
        params = self.params_dict()
        
        if train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict(),
                params=params
            )
            save_name = f"probes.pth"
        else:
            state = dict(
                epoch=epoch, 
                model=self.model.state_dict(), 
                optimizer=self.optimizer.state_dict(),
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict(),
                params=params
            )
            save_name = f"model.pth"
        ch.save(state, self.log_folder / save_name)
    
    @param('logging.log_level')
    @param('training.base_lr')
    @param('training.end_lr_ratio')
    @param('training.clip_grad')
    @param('training.use_amp')
    def train_loop(self, epoch, log_level, base_lr, end_lr_ratio, clip_grad, use_amp):
        """
            Main training loop for training with SSL or Supervised criterion.
        """
        model = self.model
        model.train()
        losses = []

        iterator = tqdm(self.train_loader)
        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):

            # Get lr
            lr = learning_schedule(
                global_step=ix,
                batch_size=self.batch_size * self.world_size,
                base_lr=base_lr,
                end_lr_ratio=end_lr_ratio,
                total_steps=self.max_steps,
                warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
            )
            for g in self.optimizer.param_groups:
                 g["lr"] = lr

            # Get data
            images_big_0 = loaders[0]
            labels_big = loaders[1]
            batch_size = loaders[1].size(0)
            if len(loaders) > 2:
                images_big_1 = loaders[2]
                images_big = ch.cat((images_big_0, images_big_1), dim=0)
            else:
                # supervised
                images_big = images_big_0

            # SSL Training
            if self.do_network_training:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss_train = ch.tensor(0.).to(self.gpu)
                with autocast(enabled=bool(use_amp)):
                    if self.teacher_student:
                        with ch.no_grad():
                            teacher_outputs, _ = self.teacher(images_big)
                            teacher_outputs = [teacher_output.view(2, batch_size, -1) for teacher_output in teacher_outputs]
                        embeddings_big, _, _ = model(images_big, predictor=True)
                    else:
                        # Compute embedding in bigger crops
                        embeddings_big, _, _ = model(images_big)
                    
                    # Compute Loss
                    step_num = 0
                    total_step_weight = 0
                    if self.teacher_student:   # byol
                        for embedding_big, teacher_output in zip(embeddings_big, teacher_outputs):
                            step_weight = self.get_loss_weighting(step_num+1)
                            total_step_weight += step_weight
                            embedding_big = embedding_big.view(2, batch_size, -1)
                            total_loss_train = total_loss_train + step_weight*self.ssl_loss(embedding_big, teacher_output)
                            step_num = step_num+1
                    elif self.supervised_loss: # category supervision
                        for embedding_big in embeddings_big:
                            step_weight = self.get_loss_weighting(step_num+1)
                            total_step_weight += step_weight
                            #we want to train exactly the same architecture with supervision, not add a classifier layer!
                            #output_classif_projector = model.module.fc(embedding_big)
                            #total_loss_train = total_loss_train + self.classif_loss(output_classif_projector, labels_big.repeat(2))
                            total_loss_train = total_loss_train + step_weight*self.classif_loss(embedding_big, labels_big)
                            step_num = step_num+1
                    else: # simclr, barlow, vicreg, ipcl
                        for embedding_big in embeddings_big:
                            step_weight = self.get_loss_weighting(step_num+1)
                            total_step_weight += step_weight
                            embedding_big = embedding_big.view(2, batch_size, -1)
                            if "simclr" in self.loss_name:
                                loss_num, loss_denum = self.ssl_loss(embedding_big[0], embedding_big[1])
                                total_loss_train = total_loss_train + step_weight*(loss_num + loss_denum)
                            else:
                                total_loss_train = total_loss_train + step_weight*self.ssl_loss(embedding_big[0], embedding_big[1])
                            step_num = step_num+1
                    
                    total_loss_train = total_loss_train / total_step_weight
                    # scaler enabling is controlled at creation, not during use
                    self.scaler.scale(total_loss_train).backward()
                    if clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        ch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    check_scaler_scale(self.scaler)

            if self.teacher_student:
                m = self.momentum_schedule[ix]  # momentum parameter
                for param_q, param_k in zip(model.module.parameters(), self.teacher.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # ======================================================================                    
            #  Online linear probes training
            # ======================================================================
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_probes.zero_grad(set_to_none=True)
            # Compute embeddings vectors
            with ch.no_grad():
                with autocast(enabled=bool(use_amp)):
                    embeddings, list_representation, _ = model(images_big_0)
                
                # if supervised, track loss and accuracy based on embeddings
                if self.supervised_loss:
                    for p,embedding in enumerate(embeddings):                            
                        current_loss = self.sup_loss(embedding.detach(), labels_big.detach())
                        self.train_meters['loss_classif_trn_p'+str(p)](current_loss.detach())
                        for k in ['top_1_trn_p'+str(p), 'top_5_trn_p'+str(p)]:
                            self.train_meters[k](embedding.detach(), labels_big.detach())
                            
            # Train probes
            with autocast(enabled=bool(use_amp)):
                # Real value classification
                list_outputs = self.probes(list_representation)
                loss_classif = 0.
                for l in range(len(list_outputs)):
                    # Compute classif loss
                    current_loss = self.sup_loss(list_outputs[l], labels_big)
                    loss_classif = loss_classif + current_loss
                    self.train_meters['loss_classif_layer'+str(l)](current_loss.detach())
                    for k in ['top_1_layer'+str(l), 'top_5_layer'+str(l)]:
                        self.train_meters[k](list_outputs[l].detach(), labels_big)
            
            self.scaler.scale(loss_classif).backward()
            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                ch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            self.scaler.step(self.optimizer_probes)
            self.scaler.update()
            check_scaler_scale(self.scaler)

            # Logging
            if log_level > 0:
                self.train_meters['loss'](total_loss_train.detach())
                losses.append(total_loss_train.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images_big.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{total_loss_train.item():.3f}']
                    names += ['loss_c']
                    values += [f'{loss_classif.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats

    @param('training.use_amp')
    def val_loop(self, use_amp):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast(enabled=bool(use_amp)):
                for images, target in tqdm(self.val_loader):
                    embeddings, list_representation, _ = model(images)
                    list_outputs = self.probes(list_representation)
                    loss_classif = 0.
                    for l in range(len(list_outputs)):
                        # Compute classif loss
                        current_loss = self.sup_loss(list_outputs[l], target)
                        loss_classif += current_loss
                        self.val_meters['loss_classif_val_layer'+str(l)](current_loss.detach())
                        for k in ['top_1_val_layer'+str(l), 'top_5_val_layer'+str(l)]:
                            self.val_meters[k](list_outputs[l].detach(), target)
                    
                    if self.supervised_loss:
                        for p,embedding in enumerate(embeddings):                            
                            current_loss = self.sup_loss(embedding, target)
                            self.val_meters['loss_classif_val_p'+str(p)](current_loss.detach())
                            for k in ['top_1_val_p'+str(p), 'top_5_val_p'+str(p)]:
                                self.val_meters[k](embedding.detach(), target)

                            
        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        folder = folder.replace("//","/")
        self.train_meters = {
            'loss': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
            'time': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
        }

        for l in range(self.n_probe_layers):
            self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
            self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)

        self.val_meters = {}
        for l in range(self.n_probe_layers):
            self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)        
                    
        if self.rank == 0:
            if Path(folder + 'final_weights.pth').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            Path(self.log_folder).mkdir(parents=True, exist_ok=True)

            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }
            
            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)
    
    @param('logging.folder')
    @param('logging.bucket_name')
    @param('logging.bucket_subfolder')
    @param('logging.use_wandb')
    @param('logging.wandb.project')
    @param('logging.wandb.entity')
    def initialize_remote_logger(self, folder, bucket_name, bucket_subfolder, use_wandb, project=None, entity=None):
        folder = folder.replace("//","/")
        if bucket_name == '' or bucket_subfolder == '': 
            self.remote_store = None
        else:
            self.remote_store = RemoteStorage(folder, bucket_name, bucket_subfolder)
            # next two steps will test whether user has write permissions to local and remote
            self.remote_store.init_logs(verbose=True)
            self.remote_store.upload_logs(verbose=True)            
            print(f'=> Remote Storage in {self.remote_store.bucket_path}')

        if use_wandb:
            config_dict = {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
            wandb.init(config=config_dict, project=project, entity=entity)

            
    @torch.no_grad()
    def add_supervised_meters(self):
        # quick dummy eval to get number output passes for this model
        self.model.eval()                
        embeddings, _, _ = self.model(torch.rand(1,3,224,224).to(self.gpu))
        num_passes = len(embeddings)
        
        # train_meters
        for p in range(num_passes):
            self.train_meters['loss_classif_trn_p'+str(p)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.train_meters['top_1_trn_p'+str(p)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
            self.train_meters['top_5_trn_p'+str(p)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)
            
        # validation accuracy on each forward pass
        for p in range(num_passes):
            self.val_meters['loss_classif_val_p'+str(p)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.val_meters['top_1_val_p'+str(p)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_val_p'+str(p)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)
                
    @param('training.train_probes_only')
    @param('logging.use_wandb')
    def log(self, content, phase, train_probes_only, use_wandb):        
        print(f'=> Log (rank={self.rank}): {content}')
        if self.rank != 0: return        
        cur_time = time.time()
        name_file = f'log_probes_{phase}.txt' if train_probes_only else f'log_{phase}.txt'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()
        
        # stream results to remote storage for live monitoring
        if self.remote_store is not None:
            self.remote_store.upload_logs()

        # log to wandb for better live monitoring :) 
        if use_wandb:
            wandb.log(content)

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.port')
    def launch_from_args(cls, distributed, world_size, port):
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0, None, 1, 1, None)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        if args[1] is not None:
            set_current_config(args[1])
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, config, ngpus_per_node, world_size, dist_url, distributed, eval_only):
        trainer = cls(gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log(None)
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    # determine if we need to automatically set the logging folder
    new_config = {}
    if len(config['logging.folder']) == 0 or config['logging.folder'] == '/tmp/':
        new_config['logging.folder'] = f"{SAVE_DIR}/logs/{config['logging.base_fn']}" 
    config.collect(new_config)   
    print(config['logging.folder'])

    if not quiet:
        config.summary()
    return config

@param('logging.folder')
@param('logging.bucket_name')
@param('logging.bucket_subfolder')
@param('dist.ngpus')
@param('dist.nodes')
@param('dist.timeout')
@param('dist.partition')
@param('dist.account')
@param('dist.comment')
@param('dist.port')
def run_submitit(config, folder, bucket_name, bucket_subfolder, ngpus, nodes,  timeout, partition, account, comment, port):
    folder = folder.replace("//","/")
    Path(folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    # Cluster specifics: To update accordingly to your cluster
    kwargs = {}
    kwargs['slurm_comment'] = comment
    executor.update_parameters(
        mem_gb=200 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=16,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_account=account,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="workshop")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(config, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at local_dir: {folder}")
    
    if bucket_name is not None and bucket_subfolder is not None:
        url = f"{bucket_name}.s3.wasabisys.com/{bucket_subfolder}".replace("//","/")
        base_url = f"https://{url}";
        train_url = f"{base_url}/log_train.txt";
        val_url = f"{base_url}/log_val.txt";
        print(f"Remote Storage Location: s3://{bucket_name}/{bucket_subfolder}")
        print(f"Train Log: {train_url}")
        print(f"Val Log: {val_url}")

@param('dist.use_submitit')
def main(config, use_submitit):
    if use_submitit:
        run_submitit(config)
    else:
        ImageNetTrainer.launch_from_args()

if __name__ == "__main__":
    config = make_config()
    main(config)
