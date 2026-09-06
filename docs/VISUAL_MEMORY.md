# Body-attached visual episodic memory

> **Historical research record.** The Python implementation and operational
> commands described here were retired from the current tree after the native
> rich developmental controller replaced them. Reproduce this experiment from
> Git commit `0caa7ef`; the findings and design rationale below remain part of
> the research record.

`chreatures/visual_memory.py` is a private, fast-binding memory for what one
creature saw, what it did, and what followed. Its input boundary is deliberately
narrow: a raw native visual feature, the creature's motor command, its
experienced transition outcome, and model time. It has no world positions,
object IDs, object kinds, scene graph, or evaluation labels.

The generative `tree | stand` hypothesis from the first resident FOV is not an
input to this organ and is not connected to the actor. This work uses the
pinned model's native 960-dimensional image representation. Semantic text from
the 500M model remains uncertain external evidence.

## Memory operation

Each relation binds in one call:

```python
memory.bind(raw_feature, action, outcome, next_raw_feature,
            model_time=t, source="experienced")
answer = memory.recall(raw_feature, action, similarity="manhattan")
```

The source must be `experienced`, `told`, `inferred`, or `imagined`. Recall can
filter these sources and reports the source on every neighbor. Predictions are
a blend of retrieved experienced transitions and the learned
action-conditioned transition/outcome heads. The result is a predicted next
representation and descriptive affordance values such as contact or effort;
it is not an action script.

The retrieval path follows the decomposition in Millidge et al.,
[Universal Hopfield Networks](https://proceedings.mlr.press/v162/millidge22a.html):
similarity selects candidates, an exponential separation sharpens their
strengths, and weighted projection returns their next representations and
outcomes. Cosine, Euclidean, and Manhattan similarity are available and are
compared on the same held-out experiences rather than assuming one metric.

## Projection drift and path context

Every episode stores both full raw native feature vectors, the native encoder
version, and the projection version in force at binding. It does not persist a
projected retrieval key. Recall passes all stored raw keys and values through
the current projection. `replace_projection` can therefore update a learned
normalizer/projector without mixing old and new latent coordinates. It refuses
to reinterpret existing features under a different native encoder because the
source pixels would be required for that migration.

Context is a separate 24-dimensional stable recurrence driven only by prior
motor commands and experienced outcomes. Its fixed private matrices and current
state persist in the checkpoint. It never consumes world coordinates. This is
a bounded application of the recurrent position-encoding result in Whittington
et al., [Relating transformers to models and neural representations of the
hippocampal formation](https://arxiv.org/abs/2112.04035): movement history can
disambiguate locally similar observations while visual relations remain
rapidly bindable. It is not a claim that this small recurrence reproduces TEM-t
or hippocampal circuitry.

## Actual research experience

`scripts/develop_visual_memory.py collect` created 320 native MuJoCo offscreen
views from 20 paths of 16 frames each. The four physical conditions contain
two visually identical purple balls in different surrounding contexts, an
actually moved ball, and a ball partially occluded by a box. The fixed
sinusoidal forward/turn/gaze sequence is explicitly a research collection
policy, not resident behavior.

The training arrays contain body-view pixels, 320 real `retina3d` ray values,
motor commands, transition outcomes, temporal order, and provenance. Object
identity, target position, moved, and occluded flags are held in a separate
`evaluation_only` manifest section. Only the latter two booleans and scenario
names score retrieval after training; none enter the model or a memory query.

| Experience artifact | Measured value |
| --- | --- |
| Native FOV PNGs | 320 at 192×144 |
| Collection wall time | 3.9845 seconds |
| `experience.npz` | 301,827 bytes |
| Dataset SHA-256 | `c9517d2272d4e901b11bd410ef7427d30c9ebc542f30b4a87b3c8b1a8f4a8d74` |
| Local path | `runs/visual-memory/experience-v1` |
| Persvati path | `/home/ember/chreatures/runs/visual-memory/experience-v1` |

## Native feature and development run

The run uses
[`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
at commit
[`7b375e1b73b11138ff12fe22c8f2822d8fe03467`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467),
whose model card declares Apache-2.0. `get_image_features` runs natively with
Transformers 4.57.1 and Torch 2.10.0+ROCm 7.0 on persvati's AMD Radeon 890M.
The full mean-pooled 960-vector is retained. A learned 960→32 projection,
action-conditioned next-representation head, and outcome head train from the
non-held-out transition sequences.

The native pass processed all 320 views in at most 757 seconds including
startup and the subsequent shape check. SmolVLM returned 13 global/adaptive
tile rows for each image. The first check stopped before training because 4,160
tile rows did not match 320 experiences. The corrected native contract groups
those rows by their input image and mean-pools the 13 rows to one 960-vector.
The original tile tensor remains on persvati with SHA-256
`429d40185ba5efe0cbb77e112a1b14b4fa0c48636525c38cd59e4feab63b0103`;
the pooled feature array hashes to
`322c7913015630ca2aeae09b513ab8fe7ace739bc7133d91438b45415447ef52`.

During extraction the device reported 328,204,288 bytes VRAM and
17,450,401,792 bytes shared GTT in use. Persvati still had 59 GiB system memory
available at that measurement. The corrected 300-epoch projector/head run used
the actual GPU, trained on 240 transitions in 0.795 seconds, and ended at loss
1.4772. On 60 held-out transitions its learned dynamics head reached mean
next-representation cosine 0.5917 and its outcome head reached MAE 0.00885.

The comparison includes compact 16×12 RGB pooling (576 values), 320 physical
retinal rays, all 960 native features, 64 fixed evenly spaced native feature
indices, and the learned 32-vector. For each representation the report measures
visual-only and recurrent path-context retrieval with cosine, Euclidean, and
Manhattan similarities. `next_representation_cosine_mean` and experienced
outcome MAE are direct transition measures. Scenario and moved/occluded scores
use privileged metadata only for external evaluation.

Manhattan retrieval was the most consistently useful of the three similarity
functions in this run. These are its 40 held-out query results; `V` is
visual/action retrieval and `P` adds recurrent path context.

| Representation | Condition top-1 V/P | Exact-path top-1 V/P | Next cosine V/P | Outcome MAE V/P |
| --- | ---: | ---: | ---: | ---: |
| Compact pixels 576 | 1.000 / 1.000 | 0.325 / 0.300 | 0.534 / 0.519 | 0.01085 / 0.01063 |
| Physical ray retina 320 | 1.000 / 1.000 | 0.225 / 0.225 | 0.726 / 0.727 | 0.01027 / 0.01048 |
| SmolVLM fixed 64 | 0.975 / 0.975 | 0.275 / 0.325 | 0.590 / 0.597 | 0.01231 / 0.01125 |
| SmolVLM full 960 | 1.000 / 1.000 | 0.400 / 0.350 | 0.568 / 0.538 | 0.01073 / 0.01130 |
| SmolVLM learned 32 | 0.975 / 0.975 | 0.375 / 0.400 | 0.433 / 0.428 | 0.01041 / 0.01017 |

The learned 32-vector compresses the native feature 30× and path context
improves its exact-path retrieval and outcome error, but it is not the strongest
representation here. Physical retinal rays best preserve the next retrieved
state, and full native features slightly outperform the learned map on scenario
classification. Context helps the fixed-64 and learned-32 VLM keys, while it
hurts or does not change several already distinctive baselines. The organ keeps
context optional rather than claiming a universal improvement.

## Artifacts and use boundary

The complete development artifacts are at
`/home/ember/chreatures/runs/visual-memory/development-v1` on persvati. A compact
20 MiB copy of the pooled features, weights, report, and restorable memory is at
`runs/visual-memory/development-v1` locally.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `smolvlm-features.npy` | 1,228,928 | `322c7913015630ca2aeae09b513ab8fe7ace739bc7133d91438b45415447ef52` |
| `visual-weights.json` | 1,008,092 | `1171748d4895665733756d3bcf4e37ad3dc386cac5ed57843afb8dc765b0999d` |
| `research-memory.json` | 17,713,519 | `8feb4a3caf3c068c78f7eda4a36c74c12bf905f2fcdb97d467e401987d0eb454` |
| `report.json` | 15,126 | `4272cbb2178de2cfb784f5a54d7d8f3c15d322cead106c0aa774a6b26e8596c8` |

The encoder version is
`smolvlm2-500m@7b375e1b73b11138ff12fe22c8f2822d8fe03467`; the trained projection
version is `projection-f16c6fdc8b96d91d`. The research checkpoint contains 300
experienced transitions with full raw features and no evaluation geometry.

`visual-weights.json` contains immutable learned weights and both encoder
versions. `smolvlm-features.npy` contains the 320 raw native features.
`research-memory.json` is a restorable private organ checkpoint whose bound
records all say `experienced`; it belongs to an unnamed research-world subject,
not Mica, Fern, Pip, or a live resident. The checkpoint stores no evaluation
geometry.

This capability is developed and measured but is not connected to the actor.
The limited paths test rapid binding and contextual aliasing in a controlled
physical setting; they do not establish open-world recognition, causal
affordances, or durable long-horizon spatial cognition.
