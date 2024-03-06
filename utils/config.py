from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
CONFIG.model.trimap_channel = 3
CONFIG.model.checkpoint = ""
CONFIG.model.imagenet_pretrain = True
CONFIG.model.imagenet_pretrain_path = ""

# hyper-parameter for refinement
CONFIG.model.self_refine_width1 = 30
CONFIG.model.self_refine_width2 = 15

# Model -> Architecture config
CONFIG.model.arch = EasyDict({})
# definition in networks/encoders/__init__.py and networks/encoders/__init__.py
CONFIG.model.arch.name = "TOMPDFB"
# predefined for GAN structure
CONFIG.model.arch.discriminator = None

CONFIG.classifier = EasyDict({})
CONFIG.classifier.resume_checkpoint =  "checkpoints/fbce.pth"

# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.cutmask_prob = 0
CONFIG.data.workers = 0
# data path for training and validation in training phase
CONFIG.data.test_image = None
CONFIG.data.test_alpha = None
CONFIG.data.test_trimap = None
CONFIG.data.test_out = None

# Logging config
CONFIG.log = EasyDict({})
CONFIG.log.experiment_root = "."
CONFIG.log.logging_path = "./logs/stdout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "./checkpoints"
CONFIG.log.checkpoint_step = 10000


def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


