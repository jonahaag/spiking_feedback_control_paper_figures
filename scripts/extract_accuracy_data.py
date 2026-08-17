"""Convert raw per-run accuracy/target-error curves into simple long-format CSVs for plotting.

Reads the memory-mapped .dat/.pkl run data via summarize_runs.load_run and writes one row per
(regime, split, seed, chip, step) to data/dynapse/<dataset>_accuracy.csv. Run from within scripts/:

    python extract_accuracy_data.py
"""
import glob
import os

import pandas as pd

import summarize_runs as sr

DATASETS = {
    'mnist1d': {
        'dir': '../dynapse_results/mnist1d',
        # regime subdirectory name -> (output 'regime' label, skipped_initial_val)
        'regimes': {
            'frozen': ('frozen', True),
            'full': ('full', False),
            'fixed_random_feedback': ('fixed-feedback', False),
        },
    },
    'yinyang': {
        'dir': '../dynapse_results/parallel_training',
        'regimes': {
            'frozen': ('frozen', False),
            'full': ('full', False),
            'fixed_random_feedback': ('fixed-feedback', False),
        },
    },
}

OUTPUT_DIR = '../data/dynapse'


def discover_runs(dataset_dir, regime):
    """Sorted run directories under dataset_dir/regime named '<seed>_<chip>'."""
    pattern = os.path.join(dataset_dir, regime, '*_*')
    return sorted(
        p for p in glob.glob(pattern)
        if sr._LEAF_RE.match(os.path.basename(os.path.normpath(p)))
    )


def extract(dataset_dir, regimes):
    rows = []
    for subdir, (label, skipped) in regimes.items():
        for path in discover_runs(dataset_dir, subdir):
            run = sr.load_run(path, skipped_initial_val=skipped)
            for split in ('train', 'val'):
                for step, accuracy, target_error in zip(
                    run[split]['x'], run[split]['classification_accuracy'], run[split]['target_error']
                ):
                    rows.append({
                        'regime': label,
                        'split': split,
                        'seed': run['seed'],
                        'chip': run['chip'],
                        'step': int(step),
                        'accuracy': float(accuracy),
                        'target_error': float(target_error),
                    })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset, cfg in DATASETS.items():
        df = extract(cfg['dir'], cfg['regimes'])
        out_path = os.path.join(OUTPUT_DIR, f'{dataset}_accuracy.csv')
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
