#include <math.h>
#include <stddef.h>
#include <mujoco/mujoco.h>

typedef struct {
  int header_version;
  int runtime_version;
  int nq;
  int nv;
  int nu;
  int nbody;
  int njnt;
  int ngeom;
  int nsite;
  int nsensor;
  int nsensordata;
  int nkey;
  int ncon;
  double timestep;
  double time;
  double max_abs_qpos;
} FlyBodySummary;

mjModel *fly_body_load(const char *path, char *error, int error_size) {
  return mj_loadXML(path, NULL, error, error_size);
}

mjData *fly_body_make_data(const mjModel *model) { return mj_makeData(model); }

void fly_body_delete_model(mjModel *model) { mj_deleteModel(model); }
void fly_body_delete_data(mjData *data) { mj_deleteData(data); }

int fly_body_startup(mjModel *model, mjData *data, int steps,
                     FlyBodySummary *out) {
  if (!model || !data || !out || steps < 0) return 0;
  if (model->nkey > 0) mj_resetDataKeyframe(model, data, 0);
  else mj_resetData(model, data);
  mj_forward(model, data);
  for (int i = 0; i < steps; ++i) mj_step(model, data);

  double max_abs = 0;
  for (int i = 0; i < model->nq; ++i) {
    if (!isfinite(data->qpos[i])) return 0;
    double value = fabs(data->qpos[i]);
    if (value > max_abs) max_abs = value;
  }
  for (int i = 0; i < model->nv; ++i) {
    if (!isfinite(data->qvel[i])) return 0;
  }
  for (int i = 0; i < model->nu; ++i) {
    if (!isfinite(data->ctrl[i])) return 0;
  }
  for (int i = 0; i < model->nsensordata; ++i) {
    if (!isfinite(data->sensordata[i])) return 0;
  }

  out->header_version = mjVERSION_HEADER;
  out->runtime_version = mj_version();
  out->nq = model->nq;
  out->nv = model->nv;
  out->nu = model->nu;
  out->nbody = model->nbody;
  out->njnt = model->njnt;
  out->ngeom = model->ngeom;
  out->nsite = model->nsite;
  out->nsensor = model->nsensor;
  out->nsensordata = model->nsensordata;
  out->nkey = model->nkey;
  out->ncon = data->ncon;
  out->timestep = model->opt.timestep;
  out->time = data->time;
  out->max_abs_qpos = max_abs;
  return 1;
}

