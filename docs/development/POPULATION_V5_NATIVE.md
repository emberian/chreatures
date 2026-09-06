# Population-v5 native mechanics receipt

The population-v5 controller adds a shared physiology adapter to the worker's
pre-GRU affine and separate active/magnitude heads for the four new rectified
actuators. The retained reference exercises both additions with nonzero weights.

This is a historical v5 mechanics receipt. The current runtime is v6; the v5
probe remains in [its frozen source revision](https://github.com/emberian/chreatures/blob/9ad2139a229887dc7302901ea793c6964128ad08/scripts/probe_native_population_v5.py)
and must be replayed with that experiment's original v5 wrapper/source and exact
external fixture and M2 binary. It is not a current-runtime acceptance command:

```sh
artifact=/tank/chreatures/scratch/v5-body-actuator-export-check/output/\
developmental-resident-population-v5-visible-parity.npz
python scripts/probe_native_population_v5.py \
  --artifact "$artifact" \
  --native-dir /tmp/chreatures-cognitive-population-v5-py312/python
```

The B3 two-step comparison matched Torch MAP values exactly on the four visible
actuators and matched recurrent state within `2.69e-7`; same-runtime snapshot
continuation was bit-exact. The fixture deliberately forces positive actuator
biases so all four outputs are visible. It proves tensor placement, arithmetic,
and persistence only. It is not a trained controller and provides no evidence
about learned behavior or actuator usefulness.

The paired Torch check produced nonzero L1 gradients for the physiology adapter
(`0.16046287`) and both new actuator heads (active weight/bias
`11.95721/0.95`, positive weight/bias `24.38642/1.93750`). A deterministic
organ perturbation changed new-axis active logits by at most `3.298695` while
the achieved-goal code changed by exactly zero. These are mechanics and gradient
routing checks, not training results.

The 11 MiB parity-only controller remains external. The compact 231 KiB input and
Torch reference and its authenticated receipt live under `data/training/`.
