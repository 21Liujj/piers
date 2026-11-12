"""Build README figures directly from the archived, measured result tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_REPO = Path(__file__).resolve().parents[2]
SOURCES = ['results/paper20261109/complete_table.json',
           'results/indoor_matched20261112/indoor_results.json']


def build(repo: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    outdoor = json.loads((repo / SOURCES[0]).read_text(encoding='utf-8'))
    indoor = json.loads((repo / SOURCES[1]).read_text(encoding='utf-8'))
    assert outdoor['status'] == indoor['status'] == 'complete'
    rows = []
    titles = ['DensePASS · mapped13', 'WildPASS · native8',
              'SynPASS · native22', 'SynPASS · mapped13']
    for title, row in zip(titles, outdoor['rows']):
        rows.append({'dataset': title, 'images': row['images'], 'classes': len(row['classes']),
                     'setting': 'frozen source weights',
                     'original_mIoU_pct': row['original']['mIoU_pct'],
                     'full_mIoU_pct': row['full_final_v3']['mIoU_pct'],
                     'delta_pp': row['delta_mIoU_pp'], 'delta95CI_pp': row['delta_mIoU_pp_95CI']})
    rows.append({'dataset': 'Stanford 2D-3D-S · SPAN8', 'images': indoor['n_images'], 'classes': 8,
                 'setting': 'shared indoor-adapted weights',
                 'original_mIoU_pct': indoor['arms']['original']['mIoU_pct'],
                 'full_mIoU_pct': indoor['arms']['full']['mIoU_pct'],
                 'delta_pp': indoor['delta_full_minus_original_mIoU_pp'],
                 'delta95CI_pp': indoor['paired_image_bootstrap_delta95CI_pp']})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#cbd5e1',
                         'text.color': '#172b3a', 'axes.labelcolor': '#334155',
                         'xtick.color': '#64748b', 'ytick.color': '#172b3a'})
    grey, teal = '#9aabba', '#087f8c'
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.subplots_adjust(left=.255, right=.975, top=.78, bottom=.18)
    y = np.arange(len(rows))
    ax.axhspan(3.55, 4.48, color='#edf6fc', zorder=0)
    ax.barh(y-.17, [r['original_mIoU_pct'] for r in rows], .29, color=grey, label='Original OmniSAM', zorder=3)
    ax.barh(y+.17, [r['full_mIoU_pct'] for r in rows], .29, color=teal, label='Complete I1 + I2 + I3', zorder=3)
    for index, row in enumerate(rows):
        for offset, key in [(-.17, 'original_mIoU_pct'), (.17, 'full_mIoU_pct')]:
            value = row[key]
            ax.text(value+.55, index+offset, f'{value:.2f}', ha='left', va='center', fontsize=10)
        ax.text(80, index, f"+{row['delta_pp']:.2f} pp", ha='right', va='center',
                color=teal, fontsize=12, fontweight='bold')
    ax.set_yticks(y, [r['dataset'] + '\n' + f"{r['images']} panoramas · {r['classes']} classes" for r in rows])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_xlim(0, 82); ax.set_xticks(np.arange(0, 71, 10)); ax.set_ylim(4.5, -.55)
    ax.set_xlabel('mIoU (%)', labelpad=10); ax.grid(axis='x', color='#e7edf2', zorder=0)
    ax.text(80, -.67, 'GAIN', ha='right', va='center', fontsize=10, color='#64748b')
    fig.text(.04, .94, 'Panoramic segmentation · matched comparisons', fontsize=19, fontweight='bold')
    fig.text(.04, .887, 'Outdoor: frozen source weights   |   Indoor: same adapted checkpoint and LoRA within the pair',
             fontsize=11, color='#536579')
    fig.legend(loc='lower left', bbox_to_anchor=(.245, .055), ncol=2, frameon=False)
    fig.text(.04, .025, 'Each row uses its own label space and protocol. Compare within a row; no cross-dataset average.',
             fontsize=10, color='#64748b')
    fig.savefig(target / 'results_overview.png', dpi=200, facecolor='white')
    plt.close(fig)

    wild = outdoor['rows'][1]
    assert wild['images'] == 500 and len(wild['classes']) == 8
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.075, right=.98, top=.75, bottom=.2)
    x = np.arange(8); width = .34
    before = wild['original']['per_class_IoU_pct']; after = wild['full_final_v3']['per_class_IoU_pct']
    left = ax.bar(x-width/2, before, width, color=grey, label='Original OmniSAM', zorder=3)
    right = ax.bar(x+width/2, after, width, color=teal, label='Complete I1 + I2 + I3', zorder=3)
    ax.bar_label(left, fmt='%.1f', fontsize=10, padding=3)
    for a, b in zip(left, right):
        padding = 18 if max(a.get_height(), b.get_height()) > 10 and abs(a.get_height()-b.get_height()) < 6 else 3
        ax.annotate(f'{b.get_height():.1f}', (b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0, padding), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x, wild['classes']); ax.set_ylim(0, 112); ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel('Class IoU (%)'); ax.grid(axis='y', color='#e7edf2', zorder=0)
    for tick, name in zip(ax.get_xticklabels(), wild['classes']):
        if name in ['curb', 'crosswalk', 'truck', 'bus']:
            tick.set_color(teal); tick.set_fontweight('bold')
    fig.text(.04, .94, 'WildPASS · where the improvement comes from', fontsize=19, fontweight='bold')
    fig.text(.04, .875, '500 panoramas · native 8 classes · frozen source weights · mIoU 37.56% → 57.57%',
             fontsize=11, color='#536579')
    fig.legend(loc='lower left', bbox_to_anchor=(.06, .045), ncol=2, frameon=False)
    fig.text(.04, .022, 'Highlighted categories have no dedicated channel in the original 13-class source head.',
             fontsize=10, color='#64748b')
    fig.savefig(target / 'wildpass_per_class.png', dpi=200, facecolor='white')
    plt.close(fig)
    provenance = {'generator': 'scripts/paper/readme_figures.py',
                  'source_sha256': {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in SOURCES},
                  'comparison_rows': rows,
                  'wildpass_class_IoU': [{'class': c, 'original': a, 'full': b}
                                        for c, a, b in zip(wild['classes'], before, after)],
                  'all_values': 'Measured archived results; no estimated or synthetic measurements'}
    (target / 'figure_data.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Created two README figures from verified archived tables.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    build(args.repo, args.out or args.repo / 'docs/assets/readme')
