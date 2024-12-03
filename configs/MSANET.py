net = 'MSANET'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 50
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = True

test_ckpt_path = None

exp_name = 'CLCD_BS4_epoch200/{}/LEVIRCD'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/LEVIRCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'Base',
        name = 'build_backbone'
    ),
    decoderhead = dict(
        type = 'MambNet',
        num_class = 2,
        channel_list = [64, 128, 256, 512],
        transform_feat = 128
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
        type = 'Adam',
        lr = 0.0005,
        beta1 = 0.9
    ),
    scheduler = dict(
        type = 'step',
        step_size = 3,
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
