#!/usr/bin/env python3
"""Seal a compact receipt for executed physical worlds and native GAM artifacts."""
import argparse
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np

from research.fly_ecology_atlas.prepare import sha, write
from research.fly_ecology_atlas.fit import scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--host-receipt', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.campaign / 'plan.json'
    plan = json.loads(plan_path.read_text())
    campaign_path = args.campaign / 'results/campaign-receipt.json'
    campaign = json.loads(campaign_path.read_text())
    fit_path = args.fit / 'report.json'
    fit = json.loads(fit_path.read_text())
    host = json.loads(args.host_receipt.read_text())
    if host['format'] != 'chreatures-fly-ecology-host-joined-receipt-v1' or any(host['files'][h] != plan[p] for h, p in [('runtime_mjs_sha256', 'runtime_sha256'), ('world_core_wasm_sha256', 'core_wasm_sha256'), ('world_json_sha256', 'base_fixture_sha256')]):
        raise ValueError('Host receipt does not bind the campaign engine and fixture')
    if campaign['plan_sha256'] != sha(plan_path) or fit['plan_sha256'] != sha(plan_path):
        raise ValueError('Campaign or fit provenance differs')
    if len(campaign['runs']) != campaign['requested_runs']:
        raise ValueError('Campaign did not attempt the complete fixed design')
    confirmation = None
    if args.confirmation:
        confirmation = json.loads(args.confirmation.read_text())
        if confirmation['plan_sha256'] != sha(plan_path) or confirmation['gam_report_sha256'] != sha(fit_path):
            raise ValueError('Physical confirmation belongs to another fit or plan')
    models = []
    for target, diagnostic in fit['diagnostics'].items():
        for model, result in diagnostic.get('models', {}).items():
            if result['status'] != 'native-fit-serialized-reloaded':
                continue
            artifact = args.fit / f'{target}-{model}.gam'
            if sha(artifact) != result['artifact_sha256']:
                raise ValueError('Serialized native model changed')
            models.append({'target': target, 'model': model, 'path': str(artifact), 'sha256': sha(artifact)})
        if diagnostic.get('status') == 'constant-target-not-fitted':
            heldout = np.array([r['metrics'][target] for r in fit['records'] if r['setting']['split'] == 'heldout'])
            diagnostic['heldout_mean_baseline'] = scores(np.full(len(heldout), diagnostic['value']), heldout) if len(heldout) else None
    compact_confirmations = []
    for result in confirmation['records'] if confirmation else []:
        projected = dict(result)
        setting_id = next(p['setting_id'] for p in fit['confirmation_proposals'] if p['intent'] == result['intent'])
        raw_path = args.campaign / 'confirmation-results' / f'{setting_id}--confirmation-layout.json'
        if sha(raw_path) != result['run_sha256']:
            raise ValueError('Raw confirmation changed')
        raw = json.loads(raw_path.read_text())
        if result['completed']:
            checkpoint = args.campaign / 'confirmation-results' / f'{setting_id}--confirmation-layout.checkpoint.json'
            state_sha = hashlib.sha256(checkpoint.read_bytes().rstrip(b'\n')).hexdigest()
            if state_sha != raw['provenance']['final_world_sha256']:
                raise ValueError('Actual saved confirmation checkpoint differs')
            projected['checkpoint'] = {'path': str(checkpoint), 'file_sha256': sha(checkpoint), 'state_sha256': state_sha}
        if result['observed'] is not None:
            observed = dict(result['observed'])
            conductance = np.asarray(observed.pop('route_diffusive_conductance_m3_s'))
            advection = np.asarray(observed.pop('effective_route_advection_m3_s'))
            observed['sum_edge_conductance_m3_s_by_pool'] = conductance.sum(axis=0).tolist()
            observed['maximum_absolute_route_advection_m3_s'] = float(np.max(np.abs(advection)))
            projected['observed'] = observed
        compact_confirmations.append(projected)
    native_module = importlib.import_module('gamfit._rust')
    receipt = {
        'format': 'chreatures-fly-ecology-development-study-receipt-v1',
        'status': 'actual-physical-campaign-and-native-gam-executed',
        'plan_sha256': sha(plan_path),
        'receipt_source_sha256': sha(__file__),
        'campaign_receipt_sha256': sha(campaign_path),
        'raw_gam_report_sha256': sha(fit_path),
        'analysis_recipe_sha256': fit['analysis_recipe_sha256'],
        'analysis_source_sha256': fit['source_sha256'],
        'host_receipt_sha256': sha(args.host_receipt),
        'physical_runtime_sha256': plan['runtime_sha256'],
        'physical_core_wasm_sha256': plan['core_wasm_sha256'],
        'base_fixture_sha256': plan['base_fixture_sha256'],
        'source_revision': plan['source_revision'],
        'diagnostic_motor': plan['neutral_motor'],
        'native_gam_build': fit['native_build'],
        'native_gam_module_sha256': sha(native_module.__file__),
        'worlds_requested': campaign['requested_runs'],
        'worlds_completed': len(fit['records']),
        'failed_worlds': fit['failed_runs'],
        'ticks_per_completed_world': plan['ticks'],
        'control_dt_s': plan['control_dt'],
        'campaign_wall_seconds': campaign['wall_seconds'],
        'workers': campaign['workers'],
        'diagnostics': fit['diagnostics'],
        'selected_models': fit['selected_models'],
        'native_artifacts': models,
        'confirmation_receipt_sha256': sha(args.confirmation) if args.confirmation else None,
        'physical_confirmations': compact_confirmations,
        'claim_limit': 'Actual finite colony growth with diagnostic neutral flies. No CNS was evaluated or replaced. Constant targets, failed fits and physical failures remain visible. GAM proposals are not promoted genotypes; confirmation execution alone does not establish predictive accuracy or successful diversity.',
    }
    write(args.output, receipt)
    print(json.dumps({'receipt': str(args.output), 'sha256': sha(args.output), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
