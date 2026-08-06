"""Convert the raw output/target-rate arrays for one run into a simple long-format CSV.

dynapse_results/mnist1d/full/201175_0/ has outputs.dat and measured_target.dat: per-sample,
per-class measured output firing rate and presented target firing rate (shape recorded in
trainlog.pkl -- unlike the weights_*.dat files, this shape is correct, no reshape workaround
needed). Per (batch, sample) exactly one of the 10 output classes is the "target" class for that
sample (identified from measured_target); this script computes, per output class and per batch,
the mean measured output rate on samples where that class was the target vs. samples where it
wasn't. Run from within scripts/:

    python extract_feedback_current_data.py
"""
import os

import numpy as np
import pandas as pd

from summarize_runs import _load_pickle

RUN_PATH = '../dynapse_results/mnist1d/full/201175_0'
OUTPUT_DIR = '../data/dynapse'


def _load_array(path, name):
    meta = _load_pickle(os.path.join(path, 'trainlog.pkl'))
    shape = tuple(meta['arrays'][name])
    return np.fromfile(os.path.join(path, f'{name}.dat'), dtype='float64').reshape(shape)


def extract(path):
    par = _load_pickle(os.path.join(path, 'params.pkl'))
    batch_size = par.batch_size

    outputs = _load_array(path, 'outputs')                  # (n_batches, batch_size, n_pop, n_out)
    measured_target = _load_array(path, 'measured_target')  # same shape
    n_batches, n_samples, n_pop, n_out = outputs.shape

    # Target class is presynaptic, so it's constant across pop copies -- average over pop first.
    target_class = measured_target.mean(axis=2).argmax(axis=-1)  # (n_batches, n_samples)
    is_target = target_class[:, :, None] == np.arange(n_out)     # (n_batches, n_samples, n_out)

    sum_target = np.einsum('bsc,bspc->bc', is_target.astype(float), outputs)
    count_target = is_target.sum(axis=1) * n_pop                 # (n_batches, n_out)
    target_rate = sum_target / count_target

    sum_all = outputs.sum(axis=(1, 2))                           # (n_batches, n_out)
    nontarget_rate = (sum_all - sum_target) / (n_samples * n_pop - count_target)

    step = np.arange(1, n_batches + 1) * batch_size
    return pd.DataFrame({
        'step': np.repeat(step, n_out),
        'class': np.tile(np.arange(n_out), n_batches),
        'target_rate': target_rate.ravel(),
        'nontarget_rate': nontarget_rate.ravel(),
    })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = extract(RUN_PATH)
    out_path = os.path.join(OUTPUT_DIR, 'mnist1d_feedback_current.csv')
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
