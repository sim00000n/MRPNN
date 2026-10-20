TRAINING = False
DEBUG = False
RESET_COUNTER = False
ALTER_LAYERS = False
PLOTTING = False
DEBUG_IMG = True
LOAD_BACKUP = False
USE_TEST_SET = True
FORCE_EVAL = False

#if not TRAINING and USE_TEST_SET:
if USE_TEST_SET:
    SET_PATH = "val_set"
else:
    SET_PATH = "ground_truth"
    #SET_PATH = "for_debug"