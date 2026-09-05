# Visitor patterns

The visitor panel records a short human performance as physical events already
supported by the 3D world: three tones, a bounded light, and motion of a free
object through the caregiver hand. It does not send a motif name, visitor
identity, or behavioral instruction into resident sensation. During replay,
residents receive the same anonymous acoustics, illumination, occlusion, and
contact physics they receive from one-off garden interactions.

Recording uses `world.time` from the authoritative state stream. Each event is
stored at an offset from the first model-time sample; changing simulation speed
therefore changes wall-clock playback speed without changing the pattern.
Pausing freezes both the world and its queue. Hand samples are coalesced by
model-time and spatial distance, all offsets round upward to the authoritative
0.05-second physics tick, and a release remains an explicit event. A motif is
bounded to 64 events and 120 model seconds.

## Garden hook

`visitor-panel.js` does not patch or wrap the garden on its own. Load its scoped
stylesheet and mount it from the garden module:

```html
<link rel="stylesheet" href="/assets/visitor-panel.css">
```

```js
import {mountVisitorPanel} from '/assets/visitor-panel.js';

let visitor;
visitor = mountVisitorPanel({
  getModelTime: () => state?.time,
  getCursor: () => ({x: cursor.x, y: cursor.y, z: Math.max(.12, cursor.z)}),
  perform: command,
});
```

After a direct physical command succeeds, offer it to the recorder. The module
ignores every operation except `signal`, `light`, `hand`, and `release`:

```js
async function command(value) {
  // Existing POST and error handling.
  const result = await sendCommand(value);
  if (result) visitor?.capture(value, state?.time);
  return result;
}
```

Feed state frames back to the display. This updates the visible model clock; it
does not execute scheduled events in the browser:

```js
state = data;
visitor?.update(data.visitor || {model_time: data.time, paused: data.paused});
```

The quick A/S/D and light buttons call the supplied, recorder-aware `perform`
hook, so the same successful-command hook captures them exactly once. Hand
paths come from the existing pointer interaction. Without `perform`, the module
acts as a pure pattern editor and adds those events without touching a world.

## Authoritative API

The default transport expects:

- `GET /api/visitor` returning `model_time`, `paused`, `revision`, `motifs`, and
  the current `queue`.
- `POST /api/visitor/motifs` accepting `{name, duration, events}` and returning
  the saved motif (optionally under `motif`).
- `POST /api/visitor/schedules` accepting either
  `{motif_id, start_in}` or a one-shot `{name, duration, events, start_in}`.
- `DELETE /api/visitor/schedules/{id}` cancelling future events. If a cancelled
  schedule still owns the hand, the backend releases it immediately. A later
  scheduled or direct hand command supersedes that ownership. Signals and
  lights already emitted retain their normal physical duration after cancel.

Event offsets and schedule times are model seconds. The server computes an
absolute `start_time` while holding the habitat lock, validates every nested
physical command with the existing world contract, and dispatches due events
from the simulation step loop. The schedule and motif archive belong in the
whole-world checkpoint alongside physics state so save/reload cannot duplicate,
skip, or shift events.

The authoritative limits are 32 saved motifs, 16 pending schedules, 64 events
per motif, and 120 model seconds per motif. The client mirrors the last two for
editing convenience; the server remains responsible for every limit.

Names are archive metadata only. The dispatch layer must construct a fresh
allowlisted command containing no schedule ID, motif ID, name, caregiver label,
or visitor identity before calling `world.command`. Scheduling and cancellation
are journalable external actions, but they are not creature memory and are not
sensory payloads.

## Standalone module check

For a UI-only preview, pass `createMemoryVisitorTransport()` as `transport` and
provide a fixed `getModelTime`/`getCursor`. This exercises recording, naming,
queue display, and cancellation without modifying a live habitat. The memory
transport is an integration fixture; it is not a persistence fallback for the
garden.

The authoritative backend now implements this queue in `visitor_events.py` and saves it with the whole world. A native articulated physical probe checkpointed inside a hand gesture, restored and replayed the remaining sound/light/hand events: physics and queue snapshots matched exactly, all five events completed once, and the hand was released.
