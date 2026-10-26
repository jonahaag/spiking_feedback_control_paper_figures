"""Convert the mnist1d population-size runs (DFA vs. Rec-LRA) into long-format CSVs for plotting.

The raw runs live (untracked) in dynapse_results/mnist1d/population_experiments/<run_set>/<timestamp>/
<seed>_<chip>/, see the README there. Each run is read via summarize_runs.load_run; the population
size and learning rates come from its params.pkl. Writes:

  * data/dynapse/mnist1d_npop2_accuracy.csv -- the n_pop=2 runs used in the figures
  * data/dynapse/mnist1d_population_archive_accuracy.csv -- every other population run, labelled by
    ``run_set``
  * dynapse_results/mnist1d/population_experiments/runs.csv -- one row per run (all run sets) with
    its settings and best-validation / test accuracy

Both accuracy CSVs have one row per (regime, n_pop, split, seed, chip, step); besides the
'train'/'val' curves, every run gets one 'test' row (at the last validation step) holding its test
scalars. Run from within scripts/:

    python extract_population_data.py
"""
import glob
import os
import pickle

import pandas as pd

import summarize_runs as sr

POPULATION_DIR = '../dynapse_results/mnist1d/population_experiments'

# run set (subdirectory of POPULATION_DIR) -> (output 'regime' label, used in figures)
# 'regime' labels match data/dynapse/mnist1d_accuracy.csv: fixed-feedback = DFA, full = Rec-LRA.
RUN_SETS = {
    'npop2_dfa': ('fixed-feedback', True),
    'npop2_rlra': ('full', True),
    'archive/sweep_lr_scaled_dfa': ('fixed-feedback', False),
    'archive/sweep_lr_scaled_rlra': ('full', False),
    'archive/npop4_unscaled_lr_dfa': ('fixed-feedback', False),
    'archive/npop4_unscaled_lr_rlra': ('full', False),
    'archive/npop2_lr2x_rlra': ('full', False),
}

LEARNING_RATES = ['learning_rate_hid1', 'learning_rate_hid2', 'learning_rate_out',
                  'learning_rate_feedback', 'learning_rate_feedback_hid1']

USED_OUTPUT_PATH = '../data/dynapse/mnist1d_npop2_accuracy.csv'
ARCHIVE_OUTPUT_PATH = '../data/dynapse/mnist1d_population_archive_accuracy.csv'
MANIFEST_PATH = os.path.join(POPULATION_DIR, 'runs.csv')

COLUMNS = ['regime', 'n_pop', 'split', 'seed', 'chip', 'step', 'accuracy', 'target_error']


def discover_runs(run_set):
    """Sorted '<timestamp>/<seed>_<chip>' run directories of one run set."""
    pattern = os.path.join(POPULATION_DIR, run_set, '*', '*_*')
    return sorted(
        p for p in glob.glob(pattern)
        if sr._LEAF_RE.match(os.path.basename(os.path.normpath(p)))
    )


def _params(path):
    with open(os.path.join(path, 'params.pkl'), 'rb') as f:
        return vars(pickle.load(f))


def extract():
    """(curve rows, manifest rows) for all run sets."""
    rows, manifest = [], []
    for run_set, (label, used) in RUN_SETS.items():
        for path in discover_runs(run_set):
            run = sr.load_run(path)
            par = _params(path)
            base = {'run_set': run_set, 'regime': label, 'n_pop': int(par['n_pop']),
                    'seed': run['seed'], 'chip': run['chip']}
            for split in ('train', 'val'):
                for step, accuracy, target_error in zip(
                    run[split]['x'], run[split]['classification_accuracy'], run[split]['target_error']
                ):
                    rows.append({**base, 'split': split, 'step': int(step),
                                 'accuracy': float(accuracy), 'target_error': float(target_error)})
            rows.append({**base, 'split': 'test', 'step': int(run['val']['x'][-1]),
                         'accuracy': run['test']['classification_accuracy'],
                         'target_error': run['test']['target_error']})
            manifest.append({
                **base, 'used': used, 'timestamp': os.path.basename(os.path.dirname(path)),
                'update_feedback_weights': par.get('update_feedback_weights'),
                **{k: par.get(k) for k in LEARNING_RATES},
                'best_val_accuracy': run['best_val']['classification_accuracy'],
                'test_accuracy': run['test']['classification_accuracy'],
            })
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def main():
    df, manifest = extract()
    used_sets = [s for s, (_, used) in RUN_SETS.items() if used]
    used = df['run_set'].isin(used_sets)
    df[used][COLUMNS].to_csv(USED_OUTPUT_PATH, index=False)
    df[~used][['run_set'] + COLUMNS].to_csv(ARCHIVE_OUTPUT_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {USED_OUTPUT_PATH} ({used.sum()} rows), {ARCHIVE_OUTPUT_PATH} ({(~used).sum()} rows), "
          f"{MANIFEST_PATH} ({len(manifest)} runs)")
    print(manifest.groupby(['run_set', 'n_pop'], sort=False).agg(
        runs=('seed', 'size'), seeds=('seed', 'nunique'),
        best_val=('best_val_accuracy', 'mean'), test=('test_accuracy', 'mean')).round(3))


if __name__ == '__main__':
    main()
