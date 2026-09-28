import math
import sys
from typing import Iterable, Optional
import logging
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tlt.data import create_token_label_target
from losses import DistillationLoss
import utils

import matplotlib.pyplot as plt
def accuracy1(pred, target, topk=(1, 5)):
    # 如果 target 是 one-hot 编码，将其转为类别索引
    if target.ndimension() > 1 and target.size(1) > 1:
        target = target.argmax(dim=1)  # 转为类别索引

    # 计算预测的类别索引
    pred = pred.argmax(dim=1)

    # 计算准确率
    correct = pred.eq(target)  # correct 是一个形状为 [batch_size] 的一维张量

    # 计算 topk 准确率
    res = []
    for k in topk:
        # 如果 correct 是一维的，就直接对它进行求和
        correct_k = correct.sum().item()  # 计算 top1 准确率
        res.append(correct_k * 100.0 / target.size(0))

    return res
# 全局变量，用于存储训练和测试的 acc1 精度
train_acc1_history = []
test_acc1_history = []

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _logger = logging.getLogger('train')
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Move data to device
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup if enabled
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        else:
            if args.token_label and args.token_label_data and not data_loader.mixup_enabled:
                targets = create_token_label_target(
                    targets,
                    num_classes=args.nb_classes,
                    smoothing=args.smoothing,
                    label_size=args.token_label_size)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)

            #print("shape ", outputs.shape) torch.Size([8, 21])

            # Compute loss
            if args.token_label:
                loss = criterion(outputs, targets)

            else: #√
                loss = criterion(samples, outputs, targets)


        # Compute Top-1 and Top-5 accuracy
        acc1, acc5 = accuracy1(outputs, targets, topk=(1, 5))
        batch_size = samples.shape[0]

        # Check if loss is finite
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            _logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward pass and optimization
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # Synchronize CUDA and update EMA model if enabled
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    _logger.info("Averaged stats:" + str(metric_logger))

    # 记录当前 epoch 的 acc1 精度
    global train_acc1_history
    train_acc1_history.append(metric_logger.acc1.global_avg)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    from sklearn.metrics import confusion_matrix
    import torch.nn.functional as F

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    _logger = logging.getLogger('train')
    header = 'Test:'

    model.eval()

    all_preds = []
    all_labels = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        preds = output.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(target.cpu())

    metric_logger.synchronize_between_processes()
    _logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    global test_acc1_history
    test_acc1_history.append(metric_logger.acc1.global_avg)

    # 返回预测标签和真实标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    } | {"preds": all_preds, "labels": all_labels}



