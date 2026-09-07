#!/usr/bin/env python3
"""Render the recorded matched-screen experiment; never synthesize neural activity."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--traces', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--preview', type=Path)
args = parser.parse_args()
raw = args.traces.read_bytes()
trace = json.loads(gzip.decompress(raw))
receipt_bytes = args.receipt.read_bytes()
receipt = json.loads(receipt_bytes)
if trace['format'] != 'chreatures-screen-response-trace-v1' or receipt['format'] != 'chreatures-screen-response-v1':
    raise ValueError('unexpected recorded experiment format')
if not receipt['matchedInitialState']['exactAcrossConditions']:
    raise ValueError('figure requires matched initial conditions')
times = trace['filmTimeSeconds']
if len(times) != receipt['execution']['ticks']:
    raise ValueError('trace and receipt lengths differ')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'chreatures-screen-v2'})
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={'hspace': .44})
fig.patch.set_facecolor('#f5f1e7')
series = [
    ('retinalChangedSites', 'Retinal sites with different RGB', 'count · resident 0', '#bd623d'),
    ('nonafferentDeltaRms', 'Non-afferent CNS response', 'RMS model difference · resident 0', '#377664'),
    ('actionDeltaRms', 'Delivered commands', 'RMS difference · 3 residents', '#376d87'),
    ('bodyDeltaRms', 'Physical body positions', 'RMS difference (m) · 3 residents', '#765a83'),
]
for ax, (key, title, units, color) in zip(axes, series):
    values = trace[key]
    if len(values) != len(times):
        raise ValueError(f'trace length differs: {key}')
    ax.set_facecolor('#f5f1e7')
    ax.plot(times, values, color=color, linewidth=1.5)
    ax.fill_between(times, values, color=color, alpha=.10)
    ax.set_ylim(bottom=0)
    ax.set_title(title + '  /  ' + units, loc='left', fontsize=10, color='#223a30', pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#bdc5b9')
    ax.grid(axis='y', color='#dce0d5', linewidth=.6)
    ax.tick_params(colors='#526154', length=3)
    ax.axvline(1.4, color='#95a396', linewidth=.7, linestyle=':')
    ax.set_xlim(0, receipt['execution']['modelSeconds'])
axes[-1].set_xlabel('Stimulus time (seconds); each sample follows one 50 ms physical step', labelpad=9)
fig.suptitle('A physical screen changes the continuing life', x=.105, y=.966,
             ha='left', color='#20392c', fontsize=20, fontweight='bold')
fig.text(.105, .921, 'Film minus blank screen · same initial bodies, full MaleCNS state and private controller state',
         color='#526154', fontsize=10)
fig.text(.105, .028,
         'Actual 30 s V2 run through Node Dawn / Metal. Initialized controller; these are responses, not recognition.\n'
         'Retinal count threshold: 10⁻⁶. Model activity is graded, not measured firing rate. Different panels use different units.',
         color='#526154', fontsize=8)
fig.subplots_adjust(left=.105, right=.97, top=.86, bottom=.13)
metadata = {
    'Title': 'Chreatures V2 matched physical screen response',
    'Description': json.dumps({'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
                               'traces_sha256': hashlib.sha256(raw).hexdigest(),
                               'matplotlib': matplotlib.__version__}, sort_keys=True),
    'Date': None,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, metadata=metadata)
if args.preview:
    fig.savefig(args.preview, dpi=140)
print(json.dumps({'file': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(), 'traces_sha256': hashlib.sha256(raw).hexdigest()}))
