# Chreatures licensing

Copyright (C) 2026 Chreatures contributors.

Original Chreatures source code is free software: you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

The complete license is in [LICENSE](LICENSE). SPDX identifier:
`AGPL-3.0-or-later`.

This notice does not replace the licenses of third-party components or data:

- GAM retains its AGPL-3.0-or-later license.
- Universal Weave retains its Unlicense dedication; the Chreatures integration
  code is covered by this project's AGPL license.
- Vendored Three.js retains its [MIT license](web/vendor/three/LICENSE).
- MaleCNS data retain their CC BY 4.0 license and attribution; see
  [the data manifest](data/malecns/manifest.json).
  The public [neuron annotation sidecar](research/fly_embodiment/OBSERVER_ANNOTATIONS.md)
  compresses and reorders those annotations into canonical graph order and adds
  explicitly engineered model-port memberships; the underlying data remain CC BY 4.0.
- The compact female FlyWire subset retains the separately documented
  CC BY-NC 4.0 restriction; see [CONNECTOME.md](docs/CONNECTOME.md).
- Pretrained model weights, other scientific data, and derived artifacts retain
  the licenses and provenance identified in their accompanying manifests and
  upstream model cards. The code license does not relicense those assets.

The public Bad Apple research video and its poster include a silent 30-second
visual excerpt from the [official Alstroemeria Records upload](https://www.youtube.com/watch?v=i41KoE0iMYU).
Shadow animation: あにら (Anira), original niconico sm8628149. Music credits:
Masayoshi Minoshima / Alstroemeria Records, nomico, and original composition by
ZUN. The visual excerpt is not relicensed under AGPL. Source hashes, processing
and credits are recorded in [the public provenance](site/assets/bad-apple-provenance.json).

The live tab vendors the official Google DeepMind MuJoCo 3.12.0 JavaScript/Wasm
runtime under [Apache-2.0](native/browser-world/licenses/MuJoCo-LICENSE). Its version and file
hashes are recorded in the physical manifest inside [the selected release bundle](site/live/release-selection.json).
The live physical screen uses the same attributed silent Bad Apple excerpt;
placing it on an interactive surface does not relicense the animation.

The anatomical body assets and author kinematics derive from NeuroMechFly /
FlyGym v2.1.0, pinned at `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`,
copyright 2023–2026 The NeuroMechFly v2 Authors, under Apache-2.0. See
[the body source notice](native/fly-body/THIRD_PARTY.md) for the retained license,
source configuration and distinctions between morphology, authored joint axes,
engineered actuation and the cross-specimen pairing with MaleCNS.
