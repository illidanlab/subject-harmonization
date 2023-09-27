import ml_collections
import os


def get_default_configs():
  
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.epochs = 100
    training.batch_size = 512
    training.log_freq = 1000
    training.valid_rate = 0.0
    training.num_workers = 16
    training.m_pretrained = True
    training.lr_init = 1.0e-3

    # finetuning
    config.finetuning = finetuning = ml_collections.ConfigDict()
    finetuning.epochs = 25
    finetuning.batch_size = 512
    finetuning.log_freq = 25
    finetuning.lr_init = 1.0e-3

    # test
    config.test = test = ml_collections.ConfigDict()
    test.batch_size = 512
    test.num_workers = 16

    # data
    config.data = data = ml_collections.ConfigDict()
    data.img_size = 32
    data.num_channels = 3
    data.num_data_points = -1
    data.num_data_points_test = -1

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.beta_min = 0.01
    model.beta_max = 2

    # model
    config.attack = attack = ml_collections.ConfigDict()
    attack.PGD = {"eps":8.0/255.0, "alpha":1.0/255.0, "steps":10}
    attack.PGDL2 = {"eps":1.0, "alpha":0.1, "steps":10}

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    return config