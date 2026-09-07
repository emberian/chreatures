#!/usr/bin/env python3
"""Actual-artifact native B8 boundary probe; synthetic receipts, not a physical skill assay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from chreatures.population import CandidateGenome
from chreatures.sensorimotor_worker_native import DevelopmentalResidentCohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--prior-artifact', type=Path, required=True)
    parser.add_argument('--assignment', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assignment = json.loads(args.assignment.read_text())
    adapter = CandidateGenome(assignment['worlds'][0]['candidates'][0]).controller_adapter()
    batch = 8
    native = DevelopmentalResidentCohort(args.artifact, batch, action_mode='map',
        goal_seed=11, action_seed=17, candidate_adapters=[copy.deepcopy(adapter) for _ in range(batch)])
    with np.load(args.artifact) as current, np.load(args.prior_artifact) as prior:
        compared = [name for name in current.files if name != 'metadata']
        assert set(compared) == set(prior.files) - {'metadata'}
        assert all(np.array_equal(current[name], prior[name]) for name in compared)
        metadata = json.loads(str(current['metadata']))
        previous_metadata = json.loads(str(prior['metadata']))
        assert metadata['consequence_laws'] == previous_metadata['consequence_laws']
    observation = np.zeros((batch, 4459), np.float32)
    neural = np.zeros((batch, 384), np.float32)
    physiology = np.tile(np.array([.82,.18,.12,0,0,.5,0,0,0,0,0,0], np.float32), (batch,1))
    previous = np.zeros((batch, 12), np.float32)
    for tick in range(10):
        observation[np.arange(batch), (tick * 37 + np.arange(batch)) % 4459] = .15
        result = native.step(observation, neural, physiology, previous,
            np.full(batch,tick,np.uint64),np.full(batch,tick*.05),np.full(batch,tick == 0))
        action = result['proposed_action'].copy()
        after = physiology.copy()
        after[:,0] -= .0002
        after[:,2] -= .0001
        after[:,3] += .001
        native.observe_consequences(np.full(batch,tick,np.uint64), physiology, after,
            action, np.full(batch,.08,np.float32), dt=.05)
        physiology, previous = after, action
    actual_snapshot = native.snapshot_value()
    # Analytical checkpoint fixture covers every remaining horizon in one B8 call.
    # Every sequence/context/outcome comes from the ten supplied execution receipts;
    # assigning cursor positions here is a shape/restore test, not learned behavior.
    fixture = copy.deepcopy(actual_snapshot)
    suffix = json.loads(fixture['native']['motor_suffix_memory'])
    for row in range(batch):
        slot = next(slot for slot in range(32) if suffix['valid'][row*32+slot]
                    and suffix['length'][row*32+slot] == 8)
        index = row * 32 + slot
        suffix['active_slot'][row] = slot
        suffix['active_generation'][row] = suffix['generation'][index]
        suffix['active_phase'][row] = row
        suffix['active_last_tick'][row] = 9 if row else None
        suffix['active_pending'][row] = False
        suffix['active_outcomes'][row*24:(row+1)*24] = [0.0]*24
        for phase in range(row):
            source = (index*8+phase)*3
            target = (row*8+phase)*3
            suffix['active_outcomes'][target:target+3] = suffix['outcomes'][source:source+3]
    fixture['native']['motor_suffix_memory'] = json.dumps(suffix)
    native = DevelopmentalResidentCohort.restore_value(fixture, args.artifact)
    restored = DevelopmentalResidentCohort.restore_value(fixture, args.artifact)
    inputs = (observation, neural, physiology, previous, np.full(batch,10,np.uint64),
              np.full(batch,.5),np.zeros(batch,bool))
    result, replay = native.step(*inputs), restored.step(*inputs)
    assert all(np.array_equal(result[name], replay[name]) for name in result)
    assert result['candidate_suffix_phase'][:,4].tolist() == list(range(8))
    assert result['candidate_suffix_length'][:,4].tolist() == list(range(8,0,-1))
    assert result['forecast_physiology'].shape == (8,8,12)
    assert all(np.isfinite(result[name]).all() for name in
               ['forecast_physiology','forecast_progress','forecast_disagreement','candidate_scores'])
    # Preserve a pending proposal and consume the same authenticated receipt twice.
    pending = native.snapshot_value()
    restored = DevelopmentalResidentCohort.restore_value(pending, args.artifact)
    after = physiology.copy()
    after[:,0] -= .0002
    after[:,2] -= .0001
    after[:,3] += .001
    receipt_a = native.observe_consequences(np.full(batch,10,np.uint64), physiology, after,
        result['proposed_action'], np.full(batch,.08,np.float32), dt=.05)
    receipt_b = restored.observe_consequences(np.full(batch,10,np.uint64), physiology, after,
        result['proposed_action'], np.full(batch,.08,np.float32), dt=.05)
    assert all(np.array_equal(receipt_a[name],receipt_b[name]) for name in receipt_a)
    assert native.snapshot_value() == restored.snapshot_value()
    suffix_after = json.loads(native.snapshot_value()['native']['motor_suffix_memory'])
    receipt = {
        'format':'chreatures-native-motor-continuation-probe-v1',
        'scope':'B8 actual imported weights and native selector; synthetic sensory/outcome receipts; analytical cursor phase sweep, not physical competence',
        'artifact':str(args.artifact.resolve()),
        'artifact_file_sha256':hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        'artifact_identity':metadata['artifact_sha256'],
        'immutable_arrays_equal_to_v7':len(compared),
        'consequence_laws_equal_to_v7':True,
        'native_format':fixture['native']['format'],
        'restored_next_decision_fields':len(result),
        'restored_pending_receipt_fields':len(receipt_a),
        'pending_receipt_snapshot_exact':True,
        'candidate_phases':result['candidate_suffix_phase'][:,4].tolist(),
        'candidate_remaining_ticks':result['candidate_suffix_length'][:,4].tolist(),
        'selected_candidates':result['selected_candidate'].tolist(),
        'completed_totals_after_fixture_receipt':suffix_after['completed_total'],
        'forecast_physiology_shape':list(result['forecast_physiology'].shape),
    }
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,sort_keys=True))


if __name__ == '__main__':
    main()
