dataset_config = dict(
    type = 'WHUCD',
    data_root = 'data/WHU_CD',
    train_mode = dict(
        transform = dict(
            RandomSizeAndCrop = {"size": 256, "crop_nopad": False},
            RandomHorizontallyFlip =None,
            RandomVerticalFlip =None,
            RandomGaussianBlur =None,
        ),
        loader = dict(
            batch_size = 16,
            num_workers = 10,
            pin_memory=True,
            shuffle = True,
            drop_last = True
        ),
    ),
    
    val_mode = dict(
        transform = dict(

        ),
        loader = dict(
            batch_size = 16,
            num_workers = 10,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),

    test_mode = dict(
        transform = dict(

        ),
        loader = dict(
            batch_size = 16,
            num_workers = 10,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),
)

