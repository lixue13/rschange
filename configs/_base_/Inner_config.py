dataset_config = dict(
    type = 'InnerCD',
    data_root = 'data/DSIFN',
    train_mode = dict(
        transform = dict(
            RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
            # RandomHorizontallyFlip ={},
            # RandomVerticalFlip ={},
            # RandomGaussianBlur ={},
           

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
            # RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
            # RandomHorizontallyFlip={},
            # RandomVerticalFlip={},
            # RandomGaussianBlur={},

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
            # RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
            # RandomHorizontallyFlip={},
            # RandomVerticalFlip={},
            # RandomGaussianBlur={},

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
