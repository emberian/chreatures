// Frame the habitat's declared volume; the outside catchment is not the subject.
export function habitatView(bounds, verticalFovDegrees, aspect) {
  if (bounds.length !== 3 || bounds.some(x => !Number.isFinite(x) || x <= 0)
      || !Number.isFinite(aspect) || aspect <= 0) throw new Error('Invalid habitat view');
  const normalize = vector => {
    const length = Math.hypot(...vector);
    return vector.map(x => x / length);
  };
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const back = normalize([.85, -1.1, 1.2]);
  const right = normalize([-back[1], back[0], 0]);
  const up = [-back[2] * right[1], back[2] * right[0],
    back[0] * right[1] - back[1] * right[0]];
  const target = bounds.map(x => x / 2);
  const tanY = Math.tan(verticalFovDegrees * Math.PI / 360);
  const tanX = tanY * aspect;
  let distance = 0;
  for (const x of [0, bounds[0]]) for (const y of [0, bounds[1]]) for (const z of [0, bounds[2]]) {
    const relative = [x, y, z].map((value, i) => value - target[i]);
    const towardCamera = dot(relative, back);
    distance = Math.max(distance,
      Math.abs(dot(relative, right)) / (tanX * .86) + towardCamera,
      Math.abs(dot(relative, up)) / (tanY * .80) + towardCamera);
  }
  return {target, position: target.map((x, i) => x + distance * back[i]), distance};
}
