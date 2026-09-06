# Rich developmental run: full-transition audit

This audit covers the completed grandchild run `rich-sensorimotor-online-v2-grandchild-update40-seed20260915`: 20,480 physical ticks at 20 Hz, four fresh 256-second episodes, four worlds, six residents per world, and 491,520 resident transitions. The retained machine-readable result is `data/analysis/rich-grandchild160-behavior.json`; `scripts/analyze_rich_development.py` regenerates it directly from every telemetry packet without loading Torch.

The final learned controller was inherited from the child update-20 checkpoint. Worker, critic, goal manager, optimizer, and their learned parameters carried forward. World state, neural state, private goal/history, and RNG state began fresh. This is therefore one continued lineage and one seed, not a matched comparison or a claim of improvement.

## What the residents actually did

The physical actions were broad but not generally saturated. Across action channels, absolute clipping at or beyond 0.95 ranged from 0.17% to 1.59%; even the first episode's all-channel rate was 1.54%, falling below 0.67% by episode four. Thrust averaged 0.0044 with standard deviation 0.311, yaw 0.0143 with standard deviation 0.346, and gaze pitch -0.0650 with standard deviation 0.417. Signals and grip remained nonnegative and moderate. The evidence does not support a policy-saturation diagnosis. Telemetry contains executed, clipped actions and cannot determine whether pre-clipping actor logits saturated.

Movement remained costly. Mean effort was 0.229 per 50 ms transition and mean mechanical work was 0.0330. Effort correlated negatively with the same-step energy change (-0.394). Mean distance per resident episode stayed between 13.38 and 14.77 world units, while mean cumulative effort ranged from 933 to 1,516.

Food contact was real but intermittent. Ninety of 96 resident-episode trajectories had at least one mouth-material contact, totaling 1,853 contact steps. Total ingested mass was 28.35 across the run, concentrated in episode one (10.35) and episode four (8.39); the middle episodes obtained about half as much. The supplied oral law stayed active, averaging 0.413, and varied inversely with gut fill (correlation -0.515). These are contiguous transition counts and conserved mass, not independent meals.

Every episode was energy-negative. Mean energy change per resident was -0.441, -0.426, -0.414, and -0.448 across episodes one through four. Every one of the 96 resident-episode physical reward sums was negative, ranging from -5.79 to -0.48. Ingestion and movement therefore did not sustain the bodies over the 256-second lifecycle even though contact and acquisition occurred.

## Goal use and reward scale

The sticky achieved-history goal switched on 9.98% of transitions, consistent with its ten-step duration. Raw encoded goal distance decreased by 0.0545 per transition on average, but progress was variable (standard deviation 0.808; 1st to 99th percentile -2.27 to +2.50). This says the observations moved slightly closer to selected achieved-history codes on average. It does not show deliberate attainment, stopping, or causal use of the goal.

With coefficient 0.01, mean goal shaping was +0.000545 per transition. Mean audited physical reward was -0.000717. Thus the dense sensory-goal term was 76.1% of the magnitude of the adverse physical signal and offset much of it, even though all four episodes remained strongly energy-negative. The last-update summary shows the same qualitative mismatch: physical reward -0.000887 and goal shaping +0.000742.

## One next training change

Continue from the same update-160 checkpoint with the same bodies, world family, objective, entropy setting, and architecture, changing only `goal_progress_coefficient` from `0.01` to `0.001`. This puts the observed mean shaping contribution near 7.6% of mean absolute physical reward instead of 76.1%, while retaining a small learned-goal credit signal. Select the continuation by preregistered full-episode physical outcomes: energy change, conserved ingested mass, effort, and mouth-contact bouts, with goal distance remaining diagnostic.

A single-variable continuation is preferable to changing the oracle, architecture, or action distribution because the run already exhibits broad nonsaturated control and physical acquisition. The current evidence identifies reward-scale competition, not absence of movement or contact. The proposed coefficient has not been trained or shown to improve regulation.

## Reproduce

On the machine holding the run:

```bash
python scripts/analyze_rich_development.py \
  /tank/chreatures/runs/development/rich-sensorimotor-online-v2-grandchild-update40-seed20260915 \
  --output data/analysis/rich-grandchild160-behavior.json
```
