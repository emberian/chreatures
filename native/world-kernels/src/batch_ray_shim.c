#include <math.h>
#include <mujoco/mujoco.h>
#include <stddef.h>

enum { OPTIC_SITES = 1771, RGB_COMPONENTS = 3 };

int chreatures_optic_scene_bind(const void *model_address, int resident_count,
                                const int *head_geoms, int excluded_body) {
  const mjModel *model = (const mjModel *)model_address;
  if (!model || resident_count <= 0 || !head_geoms || excluded_body < -1 ||
      excluded_body >= model->nbody) return -1;
  for (int resident = 0; resident < resident_count; ++resident) {
    const int head = head_geoms[resident];
    if (head < 0 || head >= model->ngeom || model->geom_bodyid[head] <= 0)
      return -2;
  }
  return resident_count;
}

static int transduce_scene_hit(const mjModel *model, int geom,
                               double distance, double illumination,
                               const float *background, float *output) {
  if (geom < 0 || !isfinite(distance)) {
    output[0] = background[0];
    output[1] = background[1];
    output[2] = background[2];
    return 0;
  }
  if (geom >= model->ngeom) return -1;
  const int material = model->geom_matid[geom];
  if (material >= model->nmat) return -2;
  const float *rgba = material >= 0 ? model->mat_rgba + 4 * material
                                    : model->geom_rgba + 4 * geom;
  const double light = 0.45 + 0.55 * illumination;
  for (int channel = 0; channel < RGB_COMPONENTS; ++channel) {
    if (!isfinite(rgba[channel])) return -3;
    output[channel] = (float)fmin(1.0, (double)rgba[channel] * light);
  }
  return 0;
}

int chreatures_optic_scene_sample(
    const void *model_address, void *data_address, int resident_count,
    const int *head_geoms, const short *site_sides, int left_sites,
    const double *eye_origins, const double *ray_directions,
    const double *illumination, double maximum_range, int excluded_body,
    const float *background_rgb, double *direction_scratch,
    double *distance_output, int *geom_output, float *rgb_output) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || resident_count <= 0 || !head_geoms || !site_sides ||
      left_sites <= 0 || left_sites >= OPTIC_SITES || !eye_origins ||
      !ray_directions || !illumination || !background_rgb ||
      !direction_scratch || !distance_output || !geom_output || !rgb_output ||
      !isfinite(maximum_range) || maximum_range <= 0 || excluded_body < -1 ||
      excluded_body >= model->nbody) return -1;
  for (int site = 0; site < OPTIC_SITES; ++site) {
    if (site_sides[site] != (site < left_sites ? 1 : 2)) return -2;
  }

  for (int resident = 0; resident < resident_count; ++resident) {
    const int head = head_geoms[resident];
    if (head < 0 || head >= model->ngeom || !isfinite(illumination[resident]) ||
        illumination[resident] < 0 || illumination[resident] > 1) return -3;
    const mjtNum *rotation = data->geom_xmat + 9 * head;
    const mjtNum *center = data->geom_xpos + 3 * head;

    for (int site = 0; site < OPTIC_SITES; ++site) {
      const double *local = ray_directions + 3 * site;
      for (int axis = 0; axis < 3; ++axis) {
        direction_scratch[3 * site + axis] =
            rotation[3 * axis] * local[0] +
            rotation[3 * axis + 1] * local[1] +
            rotation[3 * axis + 2] * local[2];
      }
      geom_output[(size_t)resident * OPTIC_SITES + site] = -1;
    }

    for (int eye = 0; eye < 2; ++eye) {
      const int first = eye == 0 ? 0 : left_sites;
      const int count = eye == 0 ? left_sites : OPTIC_SITES - left_sites;
      mjtNum origin[3];
      const double *offset = eye_origins + 3 * eye;
      for (int axis = 0; axis < 3; ++axis) {
        origin[axis] = center[axis] + rotation[3 * axis] * offset[0] +
            rotation[3 * axis + 1] * offset[1] +
            rotation[3 * axis + 2] * offset[2];
      }
      mj_multiRay(model, data, origin, direction_scratch + 3 * first, NULL, 1,
                  excluded_body,
                  geom_output + (size_t)resident * OPTIC_SITES + first,
                  distance_output + (size_t)resident * OPTIC_SITES + first,
                  NULL, count, maximum_range);
    }

    for (int site = 0; site < OPTIC_SITES; ++site) {
      const size_t index = (size_t)resident * OPTIC_SITES + site;
      if (geom_output[index] < 0 || distance_output[index] < 0) {
        geom_output[index] = -1;
        distance_output[index] = INFINITY;
      }
      const int status = transduce_scene_hit(
          model, geom_output[index], distance_output[index],
          illumination[resident], background_rgb,
          rgb_output + index * RGB_COMPONENTS);
      if (status != 0) return -10 + status;
    }
  }
  return resident_count;
}
