import torch
from tangermeme.ism import saturation_mutagenesis
import pandas as pd
import matplotlib.pyplot as plt

from domino.data.domino_data import *
from scipy import stats
from scipy.stats import gaussian_kde
from domino.models.domino_models import DominoSingleOutput
from domino.utils import tools
from domino.utils.tools import seq2one_hot
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import seaborn as sns
import fanc
import fanc.plotting as fancplot
import logging
import numpy as np
import os, sys
import pysam

from Bio.Seq import Seq


def reverse_complement(one_hot_seq):
    """
    Returns reverse complement of one-hot sequence (numpy array).
    """
    reverse = np.flip(one_hot_seq, axis=0).copy()
    rc = np.flip(reverse, axis=1).copy()
    return rc

def main():
    

    out_folder = sys.argv[1]
    DNA_context_bin_width = int(sys.argv[2])
    bound_file = sys.argv[3]
    FINAL_CUTOFF = sys.argv[4]

    genome_path = '/scratch/gpfs/PRITYKIN/brendan/microc_preprocessing/annotations/dm6.fa'

    genome_fasta = pysam.FastaFile(genome_path)

    checkpoint_file = os.listdir(f'{out_folder}/checkpoint/')[0]
    model = DominoSingleOutput.load_from_checkpoint(os.path.join(f'{out_folder}/checkpoint/', checkpoint_file))
    all_bound_seq_df = pd.read_csv(f'{out_folder}/{bound_file}')
    all_bound_seq_df.drop_duplicates(subset=['chrom', 'start', 'end'], inplace=True)
    print(all_bound_seq_df.shape)

    # get sequences (forward and reverse)
    seqs = []
    DNA_context_length = DNA_context_bin_width*200
    for i, _ in all_bound_seq_df.iterrows():
        chrom = all_bound_seq_df.loc[i, 'chrom']
        start = int(all_bound_seq_df.loc[i, 'start'])
        end = int(all_bound_seq_df.loc[i, 'end'])
        start = start - DNA_context_length
        end = end + DNA_context_length + 1   # exclusive
        seq = genome_fasta.fetch(chrom, start, end).upper()
        
        fw_seq = seq2one_hot(seq)
        fw_seq_tensor = torch.permute(torch.tensor(fw_seq).float(), (1, 0))
        seqs.append(fw_seq_tensor)

        rc_seq = reverse_complement(fw_seq)
        rc_seq_tensor = torch.permute(torch.tensor(rc_seq).float(), (1, 0))
        seqs.append(rc_seq_tensor)
    
    X = torch.stack(seqs)
    print('X shape', X.shape)
    print(X)

    start, end = 3000, 3600
    X_attr = saturation_mutagenesis(model, X, start=start, end=end, device='cuda')

    attr_arr = X_attr.detach().cpu().numpy()
    print('X_attr shape', attr_arr.shape)

    np.save(f'{out_folder}/full_ohe.bound_score_gt_{FINAL_CUTOFF}.npy', arr=X)

    start, end = X.shape[2] // 2 - 300, X.shape[2] // 2 + 300    
    windowed_seq = X[:,:, start:end]
    windowed_attributions = attr_arr

    print('window_seq shape', windowed_seq.shape)
    print('window_attr shape', windowed_attributions.shape)

    np.save(f'{out_folder}/windowed_ohe.bound_score_gt_{FINAL_CUTOFF}.npy', arr=windowed_seq)
    np.save(f'{out_folder}/windowed_attributions.bound_score_gt_{FINAL_CUTOFF}.npy', arr=windowed_attributions)

if __name__ == '__main__':
    main()