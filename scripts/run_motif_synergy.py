import torch
from domino.data.domino_data import *
from domino.models.domino_models import DominoSingleOutput
import matplotlib.pyplot as plt
from tangermeme.marginalize import marginalize
from tangermeme.ersatz import substitute

import pickle as pkl
import h5py
import os

plt.rcParams['figure.dpi'] = 200
plt.rcParams['pdf.fonttype'] = '42'

import random
from tangermeme.utils import one_hot_encode

from typing import List

import random
from typing import List
import pandas as pd
import pysam
import numpy as np
from tqdm import tqdm

def fetch_random_sequences(
    genome_path: str,
    regions_df: pd.DataFrame,
    num_sequences: int,
    sequence_length: int,
    seed: int
) -> List[str]:
    """
    Sample rows without replacement from a dataframe containing columns:
    chrom, start, end.

    Fetches sequence from [start, start + sequence_length) and returns
    sequences that contain no 'N' characters.
    """
    genome_fasta = pysam.FastaFile(genome_path)

    rng = random.Random(seed)
    sequences = []

    indices = list(regions_df.index)
    rng.shuffle(indices)

    for idx in indices:
        if len(sequences) >= num_sequences:
            break

        row = regions_df.loc[idx]

        chrom = row["chrom"]
        start = int(row["start"])

        contig_length = genome_fasta.get_reference_length(chrom)

        # Skip if sequence would extend beyond chromosome boundary
        if start + sequence_length > contig_length:
            continue

        end = start + sequence_length
        sequence = genome_fasta.fetch(chrom, start, end).upper()

        if "N" not in sequence:
            sequences.append(sequence)

    genome_fasta.close()

    if len(sequences) < num_sequences:
        raise ValueError(
            f"Only found {len(sequences)} valid sequences, "
            f"but requested {num_sequences}."
        )

    return sequences


def get_consensus(ppm):
    bases = np.array(['A', 'C', 'G', 'T'])
    max_indices = np.argmax(ppm, axis=1)
    consensus = ''.join(bases[max_indices])
    return consensus


def cooperativity_score(model, X, mA, mB, min_d, max_d):
    """
    Computes cooperativity score between motif A and B. Approach is from DeepSTARR paper.
    mA: consensus sequence of motif A
    mB: consensus sequence of motif B
    min_d: minimum specified distance of start motif-B from end of motif A
    min_d: maximum specified distance of start motif-B from end of motif A
    """
    # b: array of predictions on backbone only
    # A: array of predictions with motif A inserted in the center of sequence
    b, A = marginalize(model, X, mA, device='cuda')
    b = b.squeeze(1).detach().cpu() 
    A = A.squeeze(1).detach().cpu() 

    before_scores = []
    after_scores = []
    
    for d in range(min_d, max_d):
        # insert mB into sequence X distance d from end of motif A
        X_B = substitute(X, mB, start=(X.shape[-1] // 2) + (len(mA) // 2) + d)
        B = model(X_B.to('cuda'))
        B = B.squeeze(1).detach().cpu()
        
        # insert mA into sequence with X_B distance d from end of motif A
        X_AB = substitute(X_B, mA)
        AB = model(X_AB.to('cuda'))
        AB = AB.squeeze(1).detach().cpu() 
        
        # +/- means superadditivity/subadditivity
        # cooperativity = -1 * (AB - (A + B - b))
        # score = cooperativity.median().item()
        # scores.append(score)
        before_score = (A + B - b).tolist()
        before_scores.append(before_score)
        after_score = AB.tolist()
        after_scores.append(after_score)
        
    return before_scores, after_scores

def main():

    TFMODISCO_WINDOW=600

    # ------------------------------------------------------------------------------LOAD CONSENSUS SEQUENCES---------------------------------------------------------------------------------
    out_folder = '/scratch/gpfs/PRITYKIN/brendan/domino/output/06_12_2026_final_nc14_no_mig/DominoSingleOutput_10_16_2_1_200_32_False_False_42_0.001_plateau_5_1_512_5_256_5_4_64_gelu/'
    modisco_results = h5py.File(f"{out_folder}/ism_report_1_cutoff/modisco_results_{TFMODISCO_WINDOW}bp.h5","r")

    # ADAPTED FROM https://github.com/jmschrei/tfmodisco-lite/blob/main/modiscolite/report.py#L179
    trim_threshold = 0.3
    pattern_cwms = {}
    pattern_ppms = {}
    NUM_FLANKING_BASES = 1
    names = ['pos_patterns', 'neg_patterns'] 
    for name in names:
        metacluster = modisco_results[name]
        key = lambda x: int(x[0].split("_")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            tag = '{}.{}'.format(name, pattern_name)
            
            # if num_seqlets < 30:
            #     continue
            ppm_fwd = np.array(pattern['sequence'][:])
            ppm_rev = ppm_fwd[::-1, ::-1] # reverse complement
            
            cwm_fwd = np.array(pattern['contrib_scores'][:])
            cwm_rev = cwm_fwd[::-1, ::-1] # reverse complement
            
            # base pair contribution score is sum of attribution across position
            score_fwd = np.sum(np.abs(cwm_fwd), axis=1) 
            score_rev = np.sum(np.abs(cwm_rev), axis=1)
            
            # cut off anything less than 30% of max score
            trim_thresh_fwd = np.max(score_fwd) * trim_threshold 
            trim_thresh_rev = np.max(score_rev) * trim_threshold
            
            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]
        
            # include flanking 4 bases left and right
            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - NUM_FLANKING_BASES, 0), min(np.max(pass_inds_fwd) + NUM_FLANKING_BASES + 1, len(score_fwd) + 1)
            start_rev, end_rev = max(np.min(pass_inds_rev) - NUM_FLANKING_BASES, 0), min(np.max(pass_inds_rev) + NUM_FLANKING_BASES + 1, len(score_rev) + 1)
            
            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

            trimmed_ppm_fwd = ppm_fwd[start_fwd:end_fwd]
            # trimmed_ppm_fwd = ppm_fwd
            trimmed_ppm_rev = ppm_rev[start_rev:end_rev]
            # trimmed_ppm_rev = ppm_rev
            
            pattern_cwms[tag] = {'fwd': trimmed_cwm_fwd, 'rev': trimmed_cwm_rev}
            pattern_ppms[tag] = {'fwd': trimmed_ppm_fwd, 'rev': trimmed_ppm_rev}

    modisco_results.close()

    # ------------------------------------------------------------------------------LOAD MODEL AND BACKBONE SEQUENCES---------------------------------------------------------------------------------
    model_dir = '/scratch/gpfs/PRITYKIN/brendan/domino/output/06_12_2026_final_nc14_no_mig/DominoSingleOutput_10_16_2_1_200_32_False_False_42_0.001_plateau_5_1_512_5_256_5_4_64_gelu/'
    TRAIN_df = pd.read_csv(os.path.join(model_dir, 'TRAIN_df.csv'))
    VAL_df = pd.read_csv(os.path.join(model_dir, 'VAL_df.csv'))
    TEST_df = pd.read_csv(os.path.join(model_dir, 'TEST_df.csv'))
    ALL_df = pd.concat([TRAIN_df, VAL_df, TEST_df]).query('boundary == 0').reset_index(drop=True)

    genome_path = '/scratch/gpfs/PRITYKIN/brendan/microc_preprocessing/annotations/dm6.fa'
    random_seqs = fetch_random_sequences(genome_path, ALL_df, 100, 6600, seed=10) # fetching random sequences that have no insulation 
    X = [one_hot_encode(seq).unsqueeze(0) for seq in random_seqs]
    X = torch.cat(X, axis=0).type(torch.float32)

    # load model
    model_checkpoint_path = f"{model_dir}/checkpoint/{os.listdir(model_dir + '/checkpoint')[0]}"
    model = DominoSingleOutput.load_from_checkpoint(model_checkpoint_path, activation='gelu')
    model.eval()

    # ------------------------------------------------------------------------------LOAD CONSENSUS SEQUENCES---------------------------------------------------------------------------------
    nc14_motifs = pd.read_csv('/scratch/gpfs/PRITYKIN/brendan/domino/notebook/figures_rerun/fig6_synergy/top20_nc14_motifs_for_synergy.csv')

    domino_consensus_seqs_dict = dict()
    zipped_patterns_targets = list(zip(nc14_motifs.pattern, nc14_motifs.target))
    for pattern_id, target in zipped_patterns_targets:
        domino_consensus_seqs_dict[target] = dict()
        domino_consensus_seqs_dict[target]['fwd'] = get_consensus(pattern_ppms[pattern_id]['fwd'])
        domino_consensus_seqs_dict[target]['rev'] = get_consensus(pattern_ppms[pattern_id]['rev'])

    # ------------------------------------------------------------------------------COMPUTE COOPERATIVITY SCORES---------------------------------------------------------------------------------
    pair2scores = dict()
    for motifA in nc14_motifs.target:
        for oriA in ['fwd', 'rev']:
            mA = domino_consensus_seqs_dict[motifA][oriA]
            for motifB in nc14_motifs.target:
                for oriB in ['fwd', 'rev']:
                    mB = domino_consensus_seqs_dict[motifB][oriB]
                    print(f'started ({motifA}_{oriA}, {motifB}_{oriB})')
                    before_scores, after_scores = cooperativity_score(model, X, mA, mB, 1, 151)
                    pair2scores[(f'{motifA}_{oriA}', f'{motifB}_{oriB}')] = {'before': before_scores, 'after': after_scores}
                    print(f'completed ({motifA}_{oriA}, {motifB}_{oriB})')
        # save dict after each motif A is done
        with open("/scratch/gpfs/PRITYKIN/brendan/domino/scripts/motif_synergy/new_synergy_rerun/rc_cooperativity_scores_rerun.pkl", "wb") as f:
            pkl.dump(pair2scores, f)
    
if __name__ == '__main__':
    main()

    with open('/scratch/gpfs/PRITYKIN/brendan/domino/scripts/motif_synergy/new_synergy_rerun/rc_cooperativity_scores_rerun.pkl', 'rb') as f:
        motif_cooperativity_object = pkl.load(f)
    
    motif_rc_cooperativity_scores = dict()
    for motif_pair, pair_scores in motif_cooperativity_object.items():
        before_scores = np.array(pair_scores['before'])
        after_scores = np.array(pair_scores['after'])
        rel_change_vals = ((after_scores - before_scores)/before_scores)
        rel_change_med_vals = np.median(rel_change_vals, axis=1)
        motif_rc_cooperativity_scores[motif_pair] = rel_change_med_vals
    
    with open('/scratch/gpfs/PRITYKIN/brendan/domino/scripts/motif_synergy/new_synergy_rerun/final_rc_cooperativity_scores_rerun.pkl', 'wb') as f:
        pkl.dump(motif_rc_cooperativity_scores, f)
