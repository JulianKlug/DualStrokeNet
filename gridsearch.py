import numpy as np
import os

epochs = 30
modality = 'mri'

for network_size in range(1, 128, 24):
    for batch_size in 2**np.random.uniform(0, 8, size=8):
        batch_size = int(batch_size)
        for lr in 10**np.random.uniform(-5, 1, size=4):
            os.system(f'python train_{modality}.py'
                      f' -d /home/klug/data/working_data/withDWI_all_2016_2017/standardized_data_set.npz '
                      f'--two-d -e {epochs} -c {network_size} -b {batch_size} --lr-1 {lr} '
                      f'> {modality}_gridsearch_logs/{network_size}_{batch_size}_{lr}.log')

