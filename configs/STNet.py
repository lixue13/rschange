net = 'stnet'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor1 = 'val_change_f1'
monitor2 = 'test_change_f1'

test_ckpt_path = None

exp_name = 'CLCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/CLCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'Resnet18'
        # type = 'Swin'
    ),
    decoderhead = dict(
        type = 'STNet',
        num_class = 2,
        channel_list = [64, 128, 256, 512],
        transform_feat = 128,
        # channel_list = [96, 192, 384, 768],
        # transform_feat = 192,
        layer_num = 4
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['FocalLoss', 'dice_loss'],
    loss_weight = [0.5, 0.5],
    param = dict(
        FocalLoss = dict(
            gamma=0, 
            alpha=None
        ),
        dice_loss = dict(
            eps=1e-7
        )
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'SGD',
        lr = 0.05,
        momentum=0.9,
        weight_decay = 5e-5,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'step',
        step_size = 50,
        gamma = 0.1
    )
)

metric_cfg1 = dict(
    task = 'multiclass',
    average='micro',
    num_classes = num_class, 
)

metric_cfg2 = dict(
    task = 'multiclass',
    average='none',
    num_classes = num_class, 
)