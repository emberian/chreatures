#include <math.h>
#include <stdlib.h>
#include <mujoco/mujoco.h>

// Sample one route tube with a center ray and four parallel boundary rays.
// The authored carrier body is excluded; all other static, articulated and
// subsequently grown geometry remains capable of obstructing transport.
int chreatures_regional_route_accessibility(
    const void *model_address, void *data_address, int route_count,
    int sample_count, const int *sample_route, const double *starts,
    const double *ends, const double *radii, const int *excluded_bodies,
    double *output) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || route_count < 0 || sample_count < 0 || !sample_route ||
      !starts || !ends || !radii || !excluded_bodies || !output)
    return -1;
  for (int route = 0; route < route_count; ++route) {
    if (excluded_bodies[route] < -1 || excluded_bodies[route] >= model->nbody)
      return -1;
    output[route] = 0.0;
  }
  int *totals = (int *)calloc((size_t)route_count, sizeof(int));
  if (!totals && route_count) return -1;
  for (int sample = 0; sample < sample_count; ++sample) {
    const int route = sample_route[sample];
    if (route < 0 || route >= route_count || !isfinite(radii[sample]) ||
        radii[sample] < 0.0) {
      free(totals);
      return -1;
    }
    const double *start = starts + 3 * sample;
    const double *end = ends + 3 * sample;
    double delta[3] = {end[0] - start[0], end[1] - start[1],
                       end[2] - start[2]};
    const double length = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                               delta[2] * delta[2]);
    if (!isfinite(length) || length <= 1e-12) {
      free(totals);
      return -1;
    }
    double direction[3] = {delta[0] / length, delta[1] / length,
                           delta[2] / length};
    const double axis[3] = {fabs(direction[2]) < 0.8 ? 0.0 : 1.0, 0.0,
                            fabs(direction[2]) < 0.8 ? 1.0 : 0.0};
    double u[3] = {direction[1] * axis[2] - direction[2] * axis[1],
                   direction[2] * axis[0] - direction[0] * axis[2],
                   direction[0] * axis[1] - direction[1] * axis[0]};
    const double unorm = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (int component = 0; component < 3; ++component) u[component] /= unorm;
    double v[3] = {direction[1] * u[2] - direction[2] * u[1],
                   direction[2] * u[0] - direction[0] * u[2],
                   direction[0] * u[1] - direction[1] * u[0]};
    const double offsets[5][2] = {{0.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0},
                                  {0.0, 1.0}, {0.0, -1.0}};
    for (int ray = 0; ray < 5; ++ray) {
      double origin[3];
      for (int component = 0; component < 3; ++component)
        origin[component] = start[component] +
                            radii[sample] *
                                (offsets[ray][0] * u[component] +
                                 offsets[ray][1] * v[component]);
      int geom = -1;
      double normal[3];
      const double hit = mj_ray(model, data, origin, direction, NULL, 1,
                                excluded_bodies[route], &geom, normal);
      if (hit < 0.0 || hit >= length - 1e-6) output[route] += 1.0;
      totals[route] += 1;
    }
  }
  for (int route = 0; route < route_count; ++route) {
    if (totals[route] <= 0) {
      free(totals);
      return -1;
    }
    output[route] /= (double)totals[route];
  }
  free(totals);
  return route_count;
}
