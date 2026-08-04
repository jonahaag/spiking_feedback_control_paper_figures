"""Convert hidden-layer/input validation arrays into a simple wide-format CSV.

dynapse_results/parallel_training/full/201285_<chip>/ has val_hidden.dat and
val_measured_input.dat: per-validation-checkpoint, per-sample, per-population-copy hidden-layer
firing rate and measured input rate (shape recorded in validationlog.pkl -- correct, no reshape
workaround needed). Of the 16 validation checkpoints (0 = untrained network, 15 = end of
training), CHIP_CHECKPOINTS picks only the ones each chip's figures actually use: chip 2 needs the
full 6-checkpoint progression (Figure 2d and its supplementary full-neuron version), chips 0/1
only need the final checkpoint (the supplementary cross-chip comparison). Run from within
scripts/:

    python extract_hidden_activity_data.py
"""
import os

import numpy as np
import pandas as pd

from summarize_runs import _load_pickle

RUN_DIR = '../dynapse_results/parallel_training/full/201285_{chip}'
OUTPUT_DIR = '../data'
CHIP_CHECKPOINTS = {
    0: [15],                        # -> training step 75000 (end), for the cross-chip comparison
    1: [15],                        # -> training step 75000 (end), for the cross-chip comparison
    2: [0, 1, 5, 9, 13, 15],         # -> training steps 0, 5000, 25000, 45000, 65000, 75000 (end)
}


def _load_array(path, name):
    meta = _load_pickle(os.path.join(path, 'validationlog.pkl'))
    shape = tuple(meta['arrays'][name])
    return np.fromfile(os.path.join(path, f'val_{name}.dat'), dtype='float64').reshape(shape)


def extract(path, checkpoint_indices):
    par = _load_pickle(os.path.join(path, 'params.pkl'))
    step_per_checkpoint = par.val_frequency * par.batch_size

    hidden = _load_array(path, 'hidden')                   # (n_val, val_samples, n_pop, n_hidden)
    measured_input = _load_array(path, 'measured_input')   # (n_val, val_samples, n_pop, n_in)
    n_hidden = hidden.shape[-1]

    frames = []
    for idx in checkpoint_indices:
        hidden_mean = hidden[idx].mean(axis=1)                  # (val_samples, n_hidden)
        xy_mean = measured_input[idx, :, :, :2].mean(axis=1)    # (val_samples, 2)
        df = pd.DataFrame(hidden_mean, columns=[f'hidden_{i}' for i in range(n_hidden)])
        df.insert(0, 'y', xy_mean[:, 1])
        df.insert(0, 'x', xy_mean[:, 0])
        df.insert(0, 'sample', np.arange(len(df)))
        df.insert(0, 'step', idx * step_per_checkpoint)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []
    for chip, checkpoint_indices in CHIP_CHECKPOINTS.items():
        df = extract(RUN_DIR.format(chip=chip), checkpoint_indices)
        df.insert(0, 'chip', chip)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, 'yinyang_hidden_activity.csv')
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(result)} rows)")


if __name__ == '__main__':
    main()
