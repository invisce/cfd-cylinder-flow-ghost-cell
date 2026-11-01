"""Plot completed private Stage 0 histories; never rerun CFD or overwrite figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(folders, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    hashes = None
    for folder in map(Path, folders):
        manifest = json.loads((folder/'manifest.json').read_text())
        summary = json.loads(sorted(folder.glob('summary-*.json'))[-1].read_text())
        if summary['source_sha256'] != manifest['metadata']['source_sha256']:
            raise ValueError('Manifest and completed summary differ')
        if hashes is not None and summary['source_sha256'] != hashes:
            raise ValueError('Do not plot mixed revisions as a grid comparison')
        hashes = summary['source_sha256']
        records, final = {}, None
        for path in sorted(folder.glob('history-*.jsonl')):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    records[row['step']] = tuple(row[k] for k in
                        ('time', 'cd', 'cl', 'delta_p', 'max_divergence', 'cfl'))
                    final = row
        data = np.array([records[k] for k in sorted(records)])
        s = manifest['configuration']['solver']
        label = f"{s['nx']} x {s['ny']}"
        select = data[:, 0] >= 1.
        for ax, col, title in zip(axes.flat, (1, 2, 3, None, 4, 5),
                                 ('Drag coefficient', 'Lift coefficient', 'Pressure drop',
                                  'Final surface Cp (24 probes)', 'Maximum divergence', 'CFL')):
            ax.set_title(title)
            if col is None:
                ax.plot(np.arange(24)*15, final['cp'], marker='.', label=label)
                ax.set_xlabel('Angle from upstream (degrees)')
            else:
                ax.plot(data[select, 0], data[select, col], label=label, linewidth=1.2)
                ax.set_xlabel('Nondimensional time')
                ax.axvline(manifest['configuration']['sample_start'], color='.7', linestyle=':', linewidth=.8)
            ax.grid(alpha=.2)
    axes[1, 1].set_yscale('log')
    axes[1, 1].axhline(1e-10, color='.5', linestyle='--', linewidth=.8)
    axes[1, 2].axhline(.5, color='.5', linestyle='--', linewidth=.8)
    axes[0, 0].legend()
    fig.suptitle('Cut-Cell diagnostics: physical validation remains open')
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    plot(args.runs, args.output)
