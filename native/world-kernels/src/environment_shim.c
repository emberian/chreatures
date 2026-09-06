#include <math.h>
#include <mujoco/mujoco.h>

static void transform_point(const mjData *data, int body, const double *local,
                            const double *world_offset, double *world) {
  const double *origin = data->xpos + 3 * body;
  const double *rotation = data->xmat + 9 * body;
  for (int axis = 0; axis < 3; ++axis) {
    world[axis] = origin[axis] + world_offset[axis] +
                  rotation[3 * axis] * local[0] +
                  rotation[3 * axis + 1] * local[1] +
                  rotation[3 * axis + 2] * local[2];
  }
}

static void transform_vector(const mjData *data, int body, const double *local,
                             double *world) {
  const double *rotation = data->xmat + 9 * body;
  for (int axis = 0; axis < 3; ++axis) {
    world[axis] = rotation[3 * axis] * local[0] +
                  rotation[3 * axis + 1] * local[1] +
                  rotation[3 * axis + 2] * local[2];
  }
}

int chreatures_environment_batch(
    const void *model_address, void *data_address, int sample_count,
    const int *sample_bodies, const double *sample_local,
    const double *sample_world_offset, const int *sample_profiles,
    int profile_count, const int *profile_offsets, const double *ray_directions,
    const double *ray_weights, const double *blocked_transmission,
    const double *solar_direction, double solar_direct, double solar_diffuse,
    int light_count, const int *light_bodies, const double *light_local_position,
    const double *light_local_direction, const double *light_intensity,
    const double *light_radius, const double *flash_position,
    double flash_intensity, int flash_active, const double *bounds,
    double *illumination) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || !solar_direction || sample_count < 0 ||
      profile_count <= 0 || light_count < 0) return -1;

  for (int sample = 0; sample < sample_count; ++sample) {
    const int body = sample_bodies[sample];
    const int profile = sample_profiles[sample];
    if (body < 0 || body >= model->nbody || profile < 0 ||
        profile >= profile_count) return -2;

    double point[3];
    transform_point(data, body, sample_local + 3 * sample,
                    sample_world_offset + 3 * sample, point);
    if (point[0] < 0.0 || point[0] > bounds[0] || point[1] < 0.0 ||
        point[1] > bounds[1] || point[2] < 0.0 || point[2] > bounds[2]) {
      illumination[sample] = 0.0;
      continue;
    }

    double sky_exposure = 0.0;
    for (int ray = profile_offsets[profile];
         ray < profile_offsets[profile + 1]; ++ray) {
      const double *direction = ray_directions + 3 * ray;
      double origin[3] = {
          point[0] + 1.0e-4 * direction[0],
          point[1] + 1.0e-4 * direction[1],
          point[2] + 1.0e-4 * direction[2]};
      int geom = -1;
      double normal[3];
      const double distance =
          mj_ray(model, data, origin, direction, NULL, 1, -1, &geom, normal);
      const double exposure =
          (geom < 0 || distance < 0.0) ? 1.0 : blocked_transmission[profile];
      sky_exposure += ray_weights[ray] * exposure;
    }
    double solar_origin[3] = {
        point[0] + 1.0e-4 * solar_direction[0],
        point[1] + 1.0e-4 * solar_direction[1],
        point[2] + 1.0e-4 * solar_direction[2]};
    int solar_geom = -1;
    double solar_normal[3];
    const double solar_distance =
        mj_ray(model, data, solar_origin, solar_direction, NULL, 1, -1,
               &solar_geom, solar_normal);
    const double solar_exposure =
        (solar_geom < 0 || solar_distance < 0.0)
            ? 1.0
            : blocked_transmission[profile];
    double value = solar_diffuse * sky_exposure + solar_direct * solar_exposure;

    for (int light = 0; light < light_count; ++light) {
      const int light_body = light_bodies[light];
      if (light_body < 0 || light_body >= model->nbody) return -3;
      const double zero[3] = {0.0, 0.0, 0.0};
      double source[3], direction[3];
      transform_point(data, light_body, light_local_position + 3 * light, zero,
                      source);
      transform_vector(data, light_body, light_local_direction + 3 * light,
                       direction);
      const double delta[3] = {point[0] - source[0], point[1] - source[1],
                               point[2] - source[2]};
      const double distance =
          sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
               delta[2] * delta[2]);
      double visibility = 1.0, cone = 1.0;
      if (distance >= 1.0e-8) {
        const double toward_source[3] = {-delta[0] / distance,
                                         -delta[1] / distance,
                                         -delta[2] / distance};
        int geom = -1;
        double normal[3];
        const double ray_distance =
            mj_ray(model, data, point, toward_source, NULL, 1, -1, &geom,
                   normal);
        const int hit_body = geom >= 0 ? model->geom_bodyid[geom] : -1;
        visibility = (ray_distance < 0.0 || ray_distance >= distance - 0.07 ||
                      hit_body == light_body)
                         ? 1.0
                         : 0.10;
        const double alignment = delta[0] / distance * direction[0] +
                                 delta[1] / distance * direction[1] +
                                 delta[2] / distance * direction[2];
        cone = pow(fmax(0.0, alignment), 0.35);
      }
      const double scaled_distance = distance / light_radius[light];
      value += visibility * cone * light_intensity[light] /
               (1.0 + scaled_distance * scaled_distance);
    }

    if (flash_active) {
      const double dx = point[0] - flash_position[0];
      const double dy = point[1] - flash_position[1];
      const double dz = point[2] - flash_position[2];
      const double distance = sqrt(dx * dx + dy * dy + dz * dz);
      const double scaled_distance = distance / 1.8;
      value += flash_intensity / (1.0 + scaled_distance * scaled_distance);
    }
    illumination[sample] = fmin(1.0, value);
  }
  return sample_count;
}
