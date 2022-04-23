import pandas as pd
import h5py
import os


def convert(f):
    if type(f) != h5py._hl.dataset.Dataset:
        for key in f.keys():
            convert(f[key])
    else:
        print(f.name, f.shape)

        name = f.name.split('/')[-1]
        dir = os.path.join(os.sep.join(f.name.split('/')[1:-1]))
        os.makedirs(dir, exist_ok=True)

        df = pd.DataFrame({name: f[:]})
        df.to_csv(os.path.join(dir, name + '.csv'), index=False)


f = h5py.File('./GSE157698.h5', 'r')
convert(f)
