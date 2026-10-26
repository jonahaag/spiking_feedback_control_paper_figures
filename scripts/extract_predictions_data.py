"""Convert output validation arrays for one checkpoint into a simple predictions CSV.

dynapse_results/yinyang/full/201286_2/ has val_outputs.dat: per-validation-checkpoint,
per-sample, per-population-copy output firing rate (shape recorded in validationlog.pkl --
correct, no reshape workaround needed). For CHECKPOINT (validation checkpoint index, 0-based),
this averages over the population copies and picks the predicted class per sample as the output
neuron with the highest firing rate.

There's no recorded ground-truth label for the validation set (no measured_target.dat was ever
written for any yin-yang run), and rather than use the noisy hardware-measured input rates, both
the (x, y) input position and the true label are regenerated from
datagenerator.YinYangDataGenerator using the same val_seed/val_samples/rate range as the original
run -- with isValidation=True this reproduces the exact same sample sequence (position i in
generator.inputs/labels corresponds to sample index i everywhere else). Run from within scripts/:

    python extract_predictions_data.py
"""
import os

import numpy as np
import pandas as pd

from summarize_runs import _load_pickle
from datagenerator import YinYangDataGenerator

RUN_PATH = '../dynapse_results/yinyang/full/201286_2'
OUTPUT_DIR = '../data/dynapse'
CHECKPOINT = 8


def _load_array(path, name):
    meta = _load_pickle(os.path.join(path, 'validationlog.pkl'))
    shape = tuple(meta['arrays'][name])
    return np.fromfile(os.path.join(path, f'val_{name}.dat'), dtype='float64').reshape(shape)


def extract(path):
    par = _load_pickle(os.path.join(path, 'params.pkl'))

    outputs = _load_array(path, 'outputs')                  # (n_val, val_samples, n_pop, n_out)
    outputs_mean = outputs[CHECKPOINT].mean(axis=1)          # (val_samples, n_out)
    predicted_class = outputs_mean.argmax(axis=1)

    generator = YinYangDataGenerator(
        n_samples=par.val_samples, input_frmin=par.input_frmin, input_frmax=par.input_frmax,
        target_frmin=par.target_frmin, target_frmax=par.target_frmax, seed=par.val_seed,
        isValidation=True,
    )
    true_label = generator.labels[:, 0].astype(int)
    x, y = generator.inputs[:, 0], generator.inputs[:, 1]

    return pd.DataFrame({
        'x': x,
        'y': y,
        'predicted_class': predicted_class,
        'true_label': true_label,
    })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = extract(RUN_PATH)
    out_path = os.path.join(OUTPUT_DIR, 'yinyang_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
