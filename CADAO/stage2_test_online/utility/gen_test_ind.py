import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from stage2_test_online.config import Config

"""
This script is used to generate index sequences for testing opponent policies.
"""

if __name__ == "__main__":
    TEST_MODE = ["seen", "unseen", "mix"]
    NUM_TEST_EPISODE = 2500
    SWITCH_INTERVAL = 50
    LENGTH = (NUM_TEST_EPISODE // SWITCH_INTERVAL) * 10
    SEED = 0
    np.random.seed(SEED)
    
    for mode in TEST_MODE:
        if mode == "seen":
            test_oppo_policy = Config.SEEN_OPPO_POLICY
        elif mode == "unseen":
            test_oppo_policy = Config.UNSEEN_OPPO_POLICY
        elif mode == "mix":
            test_oppo_policy = Config.SEEN_OPPO_POLICY+Config.UNSEEN_OPPO_POLICY
        
        test_oppo_indexes = np.random.randint(len(test_oppo_policy), size=LENGTH)
        
        print(test_oppo_indexes[:])
        print(test_oppo_indexes.shape)
        
        with open(f"{mode}_oppo_indexes.npy", 'wb') as f:
            np.save(f, test_oppo_indexes)