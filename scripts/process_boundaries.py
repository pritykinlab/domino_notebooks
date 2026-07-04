from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import pandas as pd
import cooler
import numpy as np
import sys
import yaml
import os

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def has_stripe_at_bound(row, clr, OFFSET=800, prefix=''):
    region = (prefix+ row['chrom'], row['start'] - OFFSET - 1, row['end'] + OFFSET)
    arr = clr.matrix(balance=True).fetch(region)
    vstripe = arr[:, len(arr)//2].mean()
    hstripe = arr[len(arr)//2, :].mean()
    return np.isnan(vstripe) and np.isnan(hstripe)

def main():
    config_file = sys.argv[1]
    config = load_config(config_file)
    THREADS = int(config['threads'])
    MODEL_DIR = config['model_dir']
    COOLER_FILE = config['cooler_file']
    BOUND_FILE = config['bound_file']
    STRIPE_OFFSET = int(config['stripe_offset'])

    df = pd.read_csv(f'{MODEL_DIR}/{BOUND_FILE}')
    rows = df.to_dict('records')
    clr = cooler.Cooler(COOLER_FILE)

    has_stripes = list()
    func = partial(has_stripe_at_bound, clr=clr, OFFSET=STRIPE_OFFSET)
    with Pool(processes=THREADS) as pool:
        for result in tqdm(pool.imap(func, rows), total=len(df)):
            has_stripes.append(result)
        
    df['has_stripe_at_bound'] = has_stripes
    df.to_csv(os.path.join(MODEL_DIR, f"final_{BOUND_FILE.replace('.csv', '')}.nan_offset=20000.stripe_offset=800.csv"))

if __name__ == '__main__':
    main()