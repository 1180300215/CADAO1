
class Config:
    WANDB = True
    RUN_OFFLINE = False
    
    PROJECT_NAME = 'PEL-PA'
    # PROJECT_NAME = 'PEL-MS'
    
    ENV_TYPE = "PA"
    # ENV_TYPE = "MS"
    
    # ---------- NOTE: environment settings ----------
    if ENV_TYPE == "PA":  # * PA
        OBS_DIM = 8 
        ACT_DIM = 5
        NUM_STEPS = 100
        K = 20
        OPPO_INDEX = [0]
        AGENT_INDEX = [1,2]
    elif ENV_TYPE == "MS":  # * MS
        OBS_DIM = 12 
        ACT_DIM = 5
        NUM_STEPS = 100    
        K = 12
        OPPO_INDEX = [0]
        AGENT_INDEX = [1]
    
    OBS_NORMALIZE = True
    AVERAGE_TOTAL_OBS = True
    
    # ---------- NOTE: policy embedding learning ----------
    SEED_PEL = 0
    EXP_ID = 'a1-l1'
    DEVICE = 'cuda:7'
    
    if ENV_TYPE == "PA":  # * PA
        NUM_ITER = 1000
        BATCH_SIZE = 128
    elif ENV_TYPE == "MS":  # * MS
        NUM_ITER = 200
        BATCH_SIZE = 128
    NUM_UPDATE_PER_ITER = 10
    CHECKPOINT_FREQ = 200                  
    ALPHA = 1.0
    LAMBDA = 1.0
    WARMUP_STEPS = 10000
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD = 0.5
    

    TEMPERATURE = 0.1
    BASE_TEMPERATURE = 0.1

    # ---------- NOTE: neural network ----------
    HIDDEN_DIM = 32
    DROPOUT = 0.1
    NUM_LAYER = 3
    NUM_HEAD = 1
    ACTIVATION_FUNC = "relu"
    
    # ---------- NOTE: dirs ----------
    if ENV_TYPE == "PA":  # * PA
        OFFLINE_DATA_PATH = '../envs/multiagent_particle_envs/data/offline_dataset_PA_5oppo_10k.pkl'
    elif ENV_TYPE == "MS":
        OFFLINE_DATA_PATH = '../envs/markov_soccer/data/offline_dataset_MS_5oppo_10k.pkl'
    
    PEL_MODEL_DIR = 'model/'


def get_config_dict():
    config = dict(vars(Config))      
    config.pop('__doc__', None)
    config.pop('__weakref__', None)
    config.pop('__dict__', None)
    config.pop('__module__', None)
    return config


if __name__ == '__main__':
    config_ = get_config_dict()
    print(config_)