# The code is largely borrowd from https://github.com/mkocabas/VIBE
# Adhere to their licence to use this script
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import importlib
from lib.core.loss import GLoTLoss
from lib.core.trainer import Trainer
from lib.core.config import parse_args, BASE_DATA_DIR
from lib.utils.utils import prepare_output_dir
from lib.models import MotionDiscriminator
from lib.dataset._loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer
from lr_scheduler import CosineAnnealingWarmupRestarts

def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed_all(cfg.SEED_VALUE)
        
    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)

    # ========= Compile Loss ========= #
    loss = GLoTLoss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W,
        vel_or_accel_2d_weight = cfg.LOSS.vel_or_accel_2d_weight,
        vel_or_accel_3d_weight = cfg.LOSS.vel_or_accel_3d_weight,
        use_accel = cfg.LOSS.use_accel
    )

    # ========= Initialize networks, optimizers and lr_schedulers ========= #
    model_module = importlib.import_module('.%s' % cfg.MODEL.MODEL_NAME, 'lib.models')
    generator = model_module.GLoT( 
        seqlen=cfg.DATASET.SEQLEN,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        n_layers=cfg.MODEL.n_layers,
        d_model=cfg.MODEL.d_model,
        num_head=cfg.MODEL.num_head,
        dropout=cfg.MODEL.dropout,
        drop_path_r=cfg.MODEL.drop_path_r,
        atten_drop=cfg.MODEL.atten_drop,
        mask_ratio=cfg.MODEL.mask_ratio,
        short_n_layers = cfg.MODEL.short_n_layers,
        short_d_model = cfg.MODEL.short_d_model,
        short_num_head = cfg.MODEL.short_num_head,
        short_dropout = cfg.MODEL.short_dropout, 
        short_drop_path_r = cfg.MODEL.short_drop_path_r,
        short_atten_drop = cfg.MODEL.short_atten_drop,
        stride_short=cfg.MODEL.stride_short,
        drop_reg_short=cfg.MODEL.drop_reg_short,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR
        ).to(cfg.DEVICE)
    logger.info(f'net: {generator}')

    net_params = sum(map(lambda x: x.numel(), generator.parameters()))
    logger.info(f'params num: {net_params}')
    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    motion_discriminator = MotionDiscriminator(
        rnn_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
        input_size=69,
        num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
        output_size=1,
        feature_pool=cfg.TRAIN.MOT_DISCR.FEATURE_POOL,
        attention_size=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.SIZE,
        attention_layers=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.LAYERS,
        attention_dropout=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.DROPOUT
    ).to(cfg.DEVICE)
    
    dis_motion_optimizer = get_optimizer(
        model=motion_discriminator,
        optim_type=cfg.TRAIN.MOT_DISCR.OPTIM,
        lr=cfg.TRAIN.MOT_DISCR.LR,
        weight_decay=cfg.TRAIN.MOT_DISCR.WD,
        momentum=cfg.TRAIN.MOT_DISCR.MOMENTUM
    )
    
    motion_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dis_motion_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        gen_optimizer,
        first_cycle_steps = cfg.TRAIN.END_EPOCH,
        max_lr=cfg.TRAIN.GEN_LR,
        min_lr=cfg.TRAIN.GEN_LR * 0.1,
        warmup_steps=cfg.TRAIN.LR_PATIENCE,
    )
    # ========= Start Training ========= #
    Trainer(
        cfg=cfg,
        data_loaders=data_loaders,
        generator=generator,
        motion_discriminator=motion_discriminator,
        criterion=loss,
        dis_motion_optimizer=dis_motion_optimizer,
        gen_optimizer=gen_optimizer,
        writer=writer,
        lr_scheduler=lr_scheduler,
        motion_lr_scheduler=motion_lr_scheduler,
        val_epoch=cfg.TRAIN.val_epoch
    ).fit()



if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
