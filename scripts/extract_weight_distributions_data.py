"""Convert weight snapshots for one run per dataset into simple long-format CSVs of individual
weight values.

Both dynapse_results/yinyang/full/201285_0/ (yin-yang) and
dynapse_results/mnist1d/full/201175_0/ (mnist1d) have per-batch weight snapshots for their forward
(and, for yin-yang, feedback) weight matrices. The mnist1d files have a known bug: the shape
recorded in trainlog.pkl includes a spurious batch_size axis that doesn't match the actual file
size (see load_weights.py) -- the real per-file shape is (n_batches, n_pre, n_post), so that axis
is dropped for those. "Beginning of training" uses params.pkl's initial_weights (the true
pre-training values); "end of training" uses the last recorded batch(/sample). Raw weights are
summed across the population (n_pop copies) each neuron is implemented with in hardware, so
they're divided by n_pop to get the effective per-copy weight. Run from within scripts/:

    python extract_weight_distributions_data.py
"""
import os

import numpy as np
import pandas as pd

from summarize_runs import _load_pickle

OUTPUT_DIR = '../data/dynapse'

# dataset -> (run path, weight_type -> (initial_weights key, .dat filename), drop spurious batch axis)
DATASETS = {
    'yinyang': {
        'run_path': '../dynapse_results/yinyang/full/201285_0',
        'weights': {
            'in_hidden': ('in_hidden', 'weights_in_hidden_fine.dat'),
            'hidden_out': ('hidden_out', 'weights_hidden_out_fine.dat'),
            'feedback': ('feedback', 'weights_feedback.dat'),
        },
        'drop_batch_axis': False,
    },
    'mnist1d': {
        'run_path': '../dynapse_results/mnist1d/full/201175_0',
        'weights': {
            'in_hid1': ('in_hid1', 'weights_in_hid1_fine.dat'),
            'hid1_hid2': ('hid1_hid2', 'weights_hid1_hid2_fine.dat'),
            'hid2_out': ('hid2_out', 'weights_hid2_out_fine.dat'),
        },
        'drop_batch_axis': True,
    },
}


def _load_array(path, dat_filename, drop_batch_axis):
    meta = _load_pickle(os.path.join(path, 'trainlog.pkl'))
    shape = tuple(meta['arrays'][dat_filename[:-len('.dat')]])
    if drop_batch_axis:
        shape = (shape[0],) + shape[2:]
    return np.fromfile(os.path.join(path, dat_filename), dtype='float64').reshape(shape)


def extract(cfg):
    path = cfg['run_path']
    par = _load_pickle(os.path.join(path, 'params.pkl'))
    n_pop = par.n_pop

    rows = []
    for weight_type, (init_key, dat_filename) in cfg['weights'].items():
        initial = np.asarray(par.initial_weights[init_key], dtype=float).ravel() / n_pop
        final_snapshots = _load_array(path, dat_filename, cfg['drop_batch_axis'])
        final = final_snapshots[-1].ravel() / n_pop if cfg['drop_batch_axis'] else final_snapshots[-1, -1].ravel() / n_pop
        for value in initial:
            rows.append({'weight_type': weight_type, 'phase': 'init', 'weight': float(value)})
        for value in final:
            rows.append({'weight_type': weight_type, 'phase': 'end', 'weight': float(value)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset, cfg in DATASETS.items():
        df = extract(cfg)
        out_path = os.path.join(OUTPUT_DIR, f'{dataset}_weight_distributions.csv')
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
