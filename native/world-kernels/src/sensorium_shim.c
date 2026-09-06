#include <math.h>
#include <mujoco/mujoco.h>
#include <stddef.h>

enum { RETINA_COMPONENTS = 4, PERIPHERAL_ELEVATIONS = 8,
       PERIPHERAL_AZIMUTHS = 32, COARSE_ELEVATIONS = 5,
       COARSE_AZIMUTHS = 16 };

int chreatures_retina_bind(const void *model_address, int resident_count,
                           const int *root_geoms, const int *head_geoms) {
  const mjModel *model = (const mjModel *)model_address;
  if (!model || resident_count <= 0 || !root_geoms || !head_geoms) return -1;
  for (int resident = 0; resident < resident_count; ++resident) {
    const int root = root_geoms[resident];
    const int head = head_geoms[resident];
    if (root < 0 || root >= model->ngeom || head < 0 || head >= model->ngeom ||
        model->geom_bodyid[head] != model->geom_bodyid[root]) return -2;
    const mjtNum extent = model->geom_size[3 * head];
    if (!isfinite(extent) || extent <= 0) return -3;
  }
  return resident_count;
}

static int transduce_hit(const mjModel *model, int geom, mjtNum distance,
                         double illumination, double maximum_range,
                         float *output) {
  if (!isfinite(distance)) return -1;
  if (distance < 0 || distance > maximum_range || geom < 0) {
    output[0] = output[1] = output[2] = output[3] = 0.0f;
    return 0;
  }
  if (geom >= model->ngeom) return -2;
  const int material = model->geom_matid[geom];
  if (material >= model->nmat) return -3;
  const float *rgba = material >= 0 ? model->mat_rgba + 4 * material
                                    : model->geom_rgba + 4 * geom;
  const double light = 0.45 + 0.55 * illumination;
  for (int channel = 0; channel < 3; ++channel) {
    if (!isfinite(rgba[channel])) return -4;
    output[channel] = (float)fmin(1.0, (double)rgba[channel] * light);
  }
  output[3] = (float)fmax(0.0, 1.0 - distance / maximum_range);
  return 0;
}

int chreatures_retina_cohort(
    const void *model_address, void *data_address, int resident_count,
    const int *root_geoms, const int *head_geoms, const double *gaze_pitch,
    const double *illumination, int ray_count, int peripheral_rays,
    const double *ray_templates, const int *coarse_elevation_offsets,
    double maximum_range, double *direction_scratch, double *distance_scratch,
    int *geom_scratch, float *coarse_output, float *rich_output) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || resident_count <= 0 || ray_count <= 0 ||
      peripheral_rays != PERIPHERAL_ELEVATIONS * PERIPHERAL_AZIMUTHS ||
      ray_count < peripheral_rays || !root_geoms || !head_geoms ||
      !gaze_pitch || !illumination || !ray_templates ||
      !coarse_elevation_offsets || !direction_scratch || !distance_scratch ||
      !geom_scratch || !coarse_output || !rich_output ||
      !isfinite(maximum_range) || maximum_range <= 0) return -1;

  for (int resident = 0; resident < resident_count; ++resident) {
    const int root = root_geoms[resident];
    const int head = head_geoms[resident];
    if (root < 0 || root >= model->ngeom || head < 0 || head >= model->ngeom ||
        model->geom_bodyid[head] != model->geom_bodyid[root] ||
        !isfinite(gaze_pitch[resident]) ||
        !isfinite(illumination[resident]) || illumination[resident] < 0 ||
        illumination[resident] > 1) return -2;

    const mjtNum *rotation = data->geom_xmat + 9 * head;
    const mjtNum *center = data->geom_xpos + 3 * head;
    const mjtNum lens_offset = model->geom_size[3 * head] + 0.004;
    mjtNum origin[3];
    for (int axis = 0; axis < 3; ++axis)
      origin[axis] = center[axis] + rotation[3 * axis] * lens_offset;

    for (int ray = 0; ray < ray_count; ++ray) {
      const double pitch_offset = ray_templates[3 * ray];
      const double yaw_cos = ray_templates[3 * ray + 1];
      const double yaw_sin = ray_templates[3 * ray + 2];
      double pitch = 0.62 * gaze_pitch[resident] + pitch_offset;
      pitch = fmax(-1.15, fmin(1.15, pitch));
      const double pitch_cos = cos(pitch);
      const double local[3] = {
        pitch_cos * yaw_cos, pitch_cos * yaw_sin, sin(pitch)
      };
      for (int axis = 0; axis < 3; ++axis) {
        direction_scratch[3 * ray + axis] =
            rotation[3 * axis] * local[0] +
            rotation[3 * axis + 1] * local[1] +
            rotation[3 * axis + 2] * local[2];
      }
      geom_scratch[ray] = -1;
      distance_scratch[ray] = -1;
    }

    // The lens is physically outside the head. Keep every body visible so
    // antennae, legs, other residents, and environmental geometry can occlude.
    mj_multiRay(model, data, origin, direction_scratch, NULL, 1, -1,
                geom_scratch, distance_scratch, NULL, ray_count, maximum_range);

    float *resident_rich =
        rich_output + (size_t)resident * ray_count * RETINA_COMPONENTS;
    for (int ray = 0; ray < ray_count; ++ray) {
      const int status = transduce_hit(
          model, geom_scratch[ray], distance_scratch[ray],
          illumination[resident], maximum_range,
          resident_rich + (size_t)ray * RETINA_COMPONENTS);
      if (status != 0) return -10 + status;
    }

    // The canonical 5x16 retina is now an explicit area pool of actual
    // peripheral measurements. It causes no additional collision queries.
    float *resident_coarse = coarse_output +
        (size_t)resident * COARSE_ELEVATIONS * COARSE_AZIMUTHS *
        RETINA_COMPONENTS;
    for (int elevation = 0; elevation < COARSE_ELEVATIONS; ++elevation) {
      const int start = coarse_elevation_offsets[elevation];
      const int end = coarse_elevation_offsets[elevation + 1];
      if (start < 0 || end <= start || end > PERIPHERAL_ELEVATIONS) return -20;
      const float scale = 1.0f / (float)((end - start) * 2);
      for (int azimuth = 0; azimuth < COARSE_AZIMUTHS; ++azimuth) {
        float *pixel = resident_coarse +
            ((size_t)elevation * COARSE_AZIMUTHS + azimuth) *
            RETINA_COMPONENTS;
        pixel[0] = pixel[1] = pixel[2] = pixel[3] = 0.0f;
        for (int source_elevation = start; source_elevation < end;
             ++source_elevation) {
          for (int source_azimuth = 2 * azimuth;
               source_azimuth < 2 * azimuth + 2; ++source_azimuth) {
            const float *source = resident_rich +
                ((size_t)source_elevation * PERIPHERAL_AZIMUTHS +
                 source_azimuth) * RETINA_COMPONENTS;
            for (int channel = 0; channel < RETINA_COMPONENTS; ++channel)
              pixel[channel] += source[channel] * scale;
          }
        }
      }
    }
  }
  return resident_count;
}
