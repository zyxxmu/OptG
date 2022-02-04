import os
import pathlib
import random
import shutil
import time
import datetime
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.logging import *
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing
from utils.schedulers import get_policy
from utils.conv_type import SparseConv

from args import args
from trainer import train, validate

import math
import data
import models

import pdb

def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)

def set_model(model, pr_target):
    masks = []
    for n, m in model.named_modules():
        if hasattr(m, 'mask'):
            mask = m.mask.view(-1).clone()
            masks.append(mask)

    all_masks = torch.cat(masks,0)
    preserve_num = int(all_masks.size(0) * (1-pr_target))
    preserve_weight, _ = torch.topk(all_masks, preserve_num)
    threshold = preserve_weight[preserve_num-1]
    pr_cfg = []
    for mask in masks:
        pr_cfg.append(torch.sum(torch.lt(mask,threshold)).item()/mask.size(0))

    i = 0
    for n, m in model.named_modules():
        if hasattr(m, 'mask'):
            m.set_prune_rate(pr_cfg[i])
            i += 1

import numpy as np

def sigmoid(x, beta):
  
    z = np.exp(-x*beta)
    sig = 1 / (1 + z)

    return sig

def main_worker(args):
    args.gpu = None

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    ensure_path(log_base_dir)
    ensure_path(ckpt_base_dir)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger = get_logger(os.path.join(log_base_dir, 'logger'+now+'.log'))

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Evaulation of a model
    if args.evaluate:
        checkpoint = torch.load(args.evaluate_model_link)
        model.load_state_dict(checkpoint["state_dict"])        
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, logger, epoch=args.start_epoch
        )
        return

    epoch_time = AverageMeter("epoch_time", ":.4f")
    validation_time = AverageMeter("validation_time", ":.4f")
    train_time = AverageMeter("train_time", ":.4f")

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        prune_rate = args.prune_rate * sigmoid(epoch-args.epochs/2, args.beta)
        set_model(model, prune_rate)
        logger.info('Setting global pruning rate of the network to {:.3f}'.format(prune_rate))

        model = set_gpu(args, model)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, logger
        )
        train_time.update((time.time() - start_train) / 60)

        count = 0
        sum_sparse = 0.0
        for n, m in model.named_modules():
            if hasattr(m, "mask"):
                sparsity, total_params = m.getSparsity()
                logger.info("epoch{} {} sparsity {}% ".format(epoch, n, sparsity))
                sum_sparse += int(((100 - sparsity) / 100) * total_params)
                count += total_params
        total_sparsity = 100 - (100 * sum_sparse / count)
        logger.info("epoch {} sparsitytotal {}".format(epoch, total_sparsity))

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, logger, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)

        end_epoch = time.time()

def set_gpu(args, model):
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location='cuda:1')
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")



def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")

    return model


def adjust_learning_rate(optimizer, epoch):
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    if epoch < args.warmup_length:
        lr = args.lr * (epoch + 1) / args.warmup_length
    mask_lr = lr * sigmoid(epoch-args.epochs/2, args.beta)
    optimizer.param_groups[0]['lr'] = mask_lr
    optimizer.param_groups[1]['lr'] = lr

    print('mask_lr: {}, para_lr: {}'.format(mask_lr, lr))

def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        mask = [v for n, v in parameters if ("mask" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("mask" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": mask,
                    "lr": args.mask_lr,
                    "weight_decay": 0
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir

if __name__ == "__main__":
    main()
