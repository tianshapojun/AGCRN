import os
import numpy as np

def load_st_dataset(dataset, logger):
    
    data = np.load(dataset)[:,:,0]

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    logger.info('Load Dataset shaped: {}'.format(data.shape))
    return data
