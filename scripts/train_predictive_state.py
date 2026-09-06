#!/usr/bin/env python3
"""Train the recurrent predictive-state organ on anonymous real rollout rows."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from chreatures.predictive_state import PredictiveSequence, PredictiveStateConfig, PredictiveStateTrainer  # noqa:E402


def segment(source: PredictiveSequence, start: int, stop: int) -> PredictiveSequence:
    reset=source.reset[start:stop].copy(); reset[0]=True
    return PredictiveSequence(source.features[start:stop],source.physiology[start:stop],
        source.actions[start:stop],reset,source.valid[start:stop]).validated()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("rollout",type=Path)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--device",default="cpu"); p.add_argument("--latent-dim",type=int,default=96)
    p.add_argument("--heldout-fraction",type=float,default=.2); args=p.parse_args()
    sequence=PredictiveSequence.from_rollout(args.rollout); time=sequence.features.shape[0]
    split=max(2,min(time-2,int(round(time*(1-args.heldout_fraction)))))
    train,heldout=segment(sequence,0,split),segment(sequence,split,time)
    config=PredictiveStateConfig(feature_dim=sequence.features.shape[2],
        physiology_dim=sequence.physiology.shape[2],action_dim=sequence.actions.shape[2],
        latent_dim=args.latent_dim)
    trainer=PredictiveStateTrainer(config,device=args.device)
    with torch.no_grad(): _,before=trainer.loss(heldout)
    history=[]
    for _ in range(args.epochs): history.append(trainer.update(train))
    with torch.no_grad(): _,after=trainer.loss(heldout)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=trainer.checkpoint(args.output/"predictive-state.pt")
    immutable=trainer.export(args.output/"predictive-state-rust.npz")
    restored=PredictiveStateTrainer.restore(args.output/"predictive-state.pt",device=args.device)
    with torch.no_grad(): _,replayed=restored.loss(heldout)
    replay_delta=max(abs(after[key]-replayed[key]) for key in after)
    replay_a=PredictiveStateTrainer.restore(args.output/"predictive-state.pt",device=args.device)
    replay_b=PredictiveStateTrainer.restore(args.output/"predictive-state.pt",device=args.device)
    replay_metrics_a=replay_a.update(heldout); replay_metrics_b=replay_b.update(heldout)
    training_replay_delta=max(float((replay_a.model.state_dict()[name]-value).abs().max())
        for name,value in replay_b.model.state_dict().items())
    report={"format":"chreatures-predictive-state-training-v1","source":{
        "path":str(args.rollout),"sha256":hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
        "time_rows":time,"residents":sequence.features.shape[1],"seconds_per_resident_assuming_macro_dt_0.25":time*.25},
        "split":{"training_rows":split,"heldout_rows":time-split,"chronological":True},
        "heldout_before":before,"heldout_after":after,"last_training":history[-1],
        "checkpoint":checkpoint,"immutable_export":immutable,
        "checkpoint_replay_metric_max_abs_delta":replay_delta,
        "checkpoint_replay_update_model_max_abs_delta":training_replay_delta,
        "checkpoint_replay_update_metric_max_abs_delta":max(abs(replay_metrics_a[key]-replay_metrics_b[key]) for key in replay_metrics_a),
        "uncertainty_semantics":"conditional diagonal residual scale (aleatoric plus model misfit), not epistemic/OOD confidence",
        "planning_support":"exp(-h/max_trained_horizon); do not use imagined states as training experience",
        "scope":"short archived rollout smoke test; not whole-life evidence"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
