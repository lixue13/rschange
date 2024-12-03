dataset_config = dict(
    type = 'DSIFN',
    data_root = 'data/DSFIN',
    train_mode = dict(
        transform = dict(
            RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip = None,
            RandomVerticalFlip = None,
            RandomGaussianBlur = None,
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = True,
            drop_last = True
        ),
    ),
    
    val_mode = dict(
        transform = dict(
            RandomSizeAndCrop={"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip={},
            RandomVerticalFlip={},
            RandomGaussianBlur={},
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),

    test_mode = dict(
        transform = dict(
            RandomSizeAndCrop={"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip={},
            RandomVerticalFlip={},
            RandomGaussianBlur={},
        ),
        loader = dict(
            batch_size = 8,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),
)
