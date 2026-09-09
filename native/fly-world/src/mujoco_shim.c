// SPDX-License-Identifier: AGPL-3.0-or-later
#include <limits.h>
#include <math.h>
#include <mujoco/mujoco.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { FLY_WORLD_MUJOCO_VERSION = 3012000, FLY_WORLD_CONTACT_STRIDE = 20 };

// Numeric array IDs. IDs 1..99 address mjData; IDs 100+ address mjModel.
enum FlyWorldNumField {
  FLY_WORLD_NUM_QPOS = 1,
  FLY_WORLD_NUM_QVEL = 2,
  FLY_WORLD_NUM_ACT = 3,
  FLY_WORLD_NUM_QACC_WARMSTART = 4,
  FLY_WORLD_NUM_CTRL = 5,
  FLY_WORLD_NUM_QFRC_ACTUATOR = 6,
  FLY_WORLD_NUM_QFRC_CONSTRAINT = 7,
  FLY_WORLD_NUM_QFRC_APPLIED = 8,
  FLY_WORLD_NUM_XFRC_APPLIED = 9,
  FLY_WORLD_NUM_MOCAP_POS = 10,
  FLY_WORLD_NUM_MOCAP_QUAT = 11,
  FLY_WORLD_NUM_USERDATA = 12,
  FLY_WORLD_NUM_XPOS = 13,
  FLY_WORLD_NUM_XQUAT = 14,
  FLY_WORLD_NUM_XMAT = 15,
  FLY_WORLD_NUM_GEOM_XPOS = 16,
  FLY_WORLD_NUM_GEOM_XMAT = 17,
  FLY_WORLD_NUM_SENSORDATA = 18,
  FLY_WORLD_NUM_TIME = 19,
  FLY_WORLD_NUM_ACTUATOR_FORCE = 20,
  FLY_WORLD_NUM_XIPOS = 21,

  FLY_WORLD_NUM_GEOM_SIZE = 100,
  FLY_WORLD_NUM_GEOM_POS = 101,
  FLY_WORLD_NUM_GEOM_QUAT = 102,
  FLY_WORLD_NUM_GEOM_RGBA = 103,
  FLY_WORLD_NUM_GEOM_FRICTION = 104,
  FLY_WORLD_NUM_ACTUATOR_FORCERANGE = 105,
  FLY_WORLD_NUM_ACTUATOR_GAINPRM = 106,
  FLY_WORLD_NUM_MAT_RGBA = 107,
  FLY_WORLD_NUM_MESH_VERT = 108,
  FLY_WORLD_NUM_MESH_NORMAL = 109,
};

enum FlyWorldIntField {
  FLY_WORLD_INT_GEOM_BODYID = 1,
  FLY_WORLD_INT_GEOM_TYPE = 2,
  FLY_WORLD_INT_GEOM_MATID = 3,
  FLY_WORLD_INT_GEOM_CONTYPE = 4,
  FLY_WORLD_INT_GEOM_CONAFFINITY = 5,
  FLY_WORLD_INT_GEOM_GROUP = 6,
  FLY_WORLD_INT_MESH_VERTADR = 7,
  FLY_WORLD_INT_MESH_VERTNUM = 8,
  FLY_WORLD_INT_MESH_NORMALADR = 9,
  FLY_WORLD_INT_MESH_NORMALNUM = 10,
  FLY_WORLD_INT_MESH_FACEADR = 11,
  FLY_WORLD_INT_MESH_FACENUM = 12,
  FLY_WORLD_INT_MESH_FACE = 13,
  FLY_WORLD_INT_GEOM_DATAID = 14,
};

typedef struct FlyWorldDimensions {
  int header_version;
  int runtime_version;
  int nq;
  int nv;
  int nu;
  int na;
  int nbody;
  int njnt;
  int ngeom;
  int nsite;
  int nsensor;
  int nsensordata;
  int nkey;
  int nmocap;
  int nuserdata;
  int neq;
  int npair;
  int nmat;
  int nmesh;
  int nmeshvert;
  int nmeshnormal;
  int nmeshface;
  int integration_state_size;
  int ncon;
  int integrator;
  double timestep;
  double time;
} FlyWorldDimensions;

static int size_to_int(mjtSize value, int *out) {
  if (!out || value < 0 || value > INT_MAX) return 0;
  *out = (int)value;
  return 1;
}

int fly_world_header_version(void) { return mjVERSION_HEADER; }
int fly_world_runtime_version(void) { return mj_version(); }
int fly_world_version_compatible(void) {
  return mjVERSION_HEADER == FLY_WORLD_MUJOCO_VERSION &&
         mj_version() == FLY_WORLD_MUJOCO_VERSION;
}

void *fly_world_load_xml(const char *path, char *error, size_t error_capacity) {
  if (!path || (error_capacity && !error)) return NULL;
  const int capacity = error_capacity > INT_MAX ? INT_MAX : (int)error_capacity;
  if (error && capacity) error[0] = '\0';
  return (void *)mj_loadXML(path, NULL, error, capacity);
}

void *fly_world_make_data(const void *model_address) {
  if (!model_address) return NULL;
  return (void *)mj_makeData((const mjModel *)model_address);
}

void fly_world_delete_data(void *data_address) {
  if (data_address) mj_deleteData((mjData *)data_address);
}

void fly_world_delete_model(void *model_address) {
  if (model_address) mj_deleteModel((mjModel *)model_address);
}

int64_t fly_world_material_albedo_len(const void *model_address) {
  const mjModel *model = (const mjModel *)model_address;
  if (!model || model->nmat < 0 || model->nmat > INT64_MAX / 3) return -1;
  return (int64_t)model->nmat * 3;
}

int fly_world_material_albedo(const void *model_address, double *output,
                              size_t count) {
  const mjModel *model = (const mjModel *)model_address;
  const int64_t expected = fly_world_material_albedo_len(model_address);
  if (expected < 0 || count != (size_t)expected || (count && !output)) return 0;
  for (int material = 0; material < model->nmat; ++material) {
    output[material * 3] = 1.0;
    output[material * 3 + 1] = 1.0;
    output[material * 3 + 2] = 1.0;
    int texture = model->mat_texid[material * mjNTEXROLE + mjTEXROLE_RGB];
    if (texture < 0) {
      texture = model->mat_texid[material * mjNTEXROLE + mjTEXROLE_RGBA];
    }
    if (texture < 0) continue;
    if (texture >= model->ntex || !model->tex_width || !model->tex_height ||
        !model->tex_nchannel || !model->tex_colorspace || !model->tex_adr ||
        !model->tex_data) return 0;
    const int width = model->tex_width[texture];
    const int height = model->tex_height[texture];
    const int channels = model->tex_nchannel[texture];
    const mjtSize address = model->tex_adr[texture];
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4 || address < 0) return 0;
    if ((size_t)width > SIZE_MAX / (size_t)height || model->ntexdata < 0) return 0;
    const size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / (size_t)channels) return 0;
    const size_t bytes = pixels * (size_t)channels;
    if ((size_t)address > (size_t)model->ntexdata ||
        bytes > (size_t)model->ntexdata - (size_t)address) return 0;
    double sum[3] = {0.0, 0.0, 0.0};
    const mjtByte *data = model->tex_data + address;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      mjtByte rgb[3];
      if (channels == 1 || channels == 2) {
        rgb[0] = rgb[1] = rgb[2] = data[pixel * channels];
      } else {
        rgb[0] = data[pixel * channels];
        rgb[1] = data[pixel * channels + 1];
        rgb[2] = data[pixel * channels + 2];
      }
      for (int channel = 0; channel < 3; ++channel) {
        if (model->tex_colorspace[texture] == mjCOLORSPACE_SRGB) {
          const double encoded = (double)rgb[channel] / 255.0;
          sum[channel] += encoded <= 0.04045
              ? encoded / 12.92
              : pow((encoded + 0.055) / 1.055, 2.4);
        } else {
          sum[channel] += (double)rgb[channel] / 255.0;
        }
      }
    }
    for (int channel = 0; channel < 3; ++channel) {
      output[material * 3 + channel] =
          sum[channel] / (double)pixels;
    }
  }
  return 1;
}

int fly_world_dimensions(const void *model_address, const void *data_address,
                         FlyWorldDimensions *out) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !out) return 0;
  memset(out, 0, sizeof(*out));
  out->header_version = mjVERSION_HEADER;
  out->runtime_version = mj_version();
#define SET_SIZE(name) if (!size_to_int(model->name, &out->name)) return 0
  SET_SIZE(nq);
  SET_SIZE(nv);
  SET_SIZE(nu);
  SET_SIZE(na);
  SET_SIZE(nbody);
  SET_SIZE(njnt);
  SET_SIZE(ngeom);
  SET_SIZE(nsite);
  SET_SIZE(nsensor);
  SET_SIZE(nsensordata);
  SET_SIZE(nkey);
  SET_SIZE(nmocap);
  SET_SIZE(nuserdata);
  SET_SIZE(neq);
  SET_SIZE(npair);
  SET_SIZE(nmat);
  SET_SIZE(nmesh);
  SET_SIZE(nmeshvert);
  SET_SIZE(nmeshnormal);
  SET_SIZE(nmeshface);
#undef SET_SIZE
  out->integration_state_size = mj_stateSize(model, mjSTATE_INTEGRATION);
  out->ncon = data ? data->ncon : 0;
  out->integrator = model->opt.integrator;
  out->timestep = model->opt.timestep;
  out->time = data ? data->time : 0.0;
  return out->integration_state_size >= 0;
}

int fly_world_reset(const void *model_address, void *data_address) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data) return 0;
  mj_resetData(model, data);
  return 1;
}

int fly_world_reset_keyframe(const void *model_address, void *data_address,
                             int key) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || key < 0 || key >= model->nkey) return 0;
  mj_resetDataKeyframe(model, data, key);
  return 1;
}

int fly_world_forward(const void *model_address, void *data_address) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data) return 0;
  mj_forward(model, data);
  return 1;
}

int fly_world_step1(const void *model_address, void *data_address) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data) return 0;
  mj_step1(model, data);
  return 1;
}

int fly_world_step2(const void *model_address, void *data_address) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data) return 0;
  mj_step2(model, data);
  return 1;
}

int fly_world_set_const(void *model_address, void *data_address) {
  mjModel *model = (mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data) return 0;
  mj_setConst(model, data);
  return 1;
}

int fly_world_state_integration(void) { return mjSTATE_INTEGRATION; }
int fly_world_state_eq_active(void) { return mjSTATE_EQ_ACTIVE; }

int fly_world_state_size(const void *model_address, int specification) {
  const mjModel *model = (const mjModel *)model_address;
  return model ? mj_stateSize(model, specification) : -1;
}

int fly_world_get_state(const void *model_address, const void *data_address,
                        int specification, double *output, size_t length) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  const int required = model ? mj_stateSize(model, specification) : -1;
  if (!model || !data || required < 0 || length != (size_t)required ||
      (length && !output)) return 0;
  mj_getState(model, data, output, specification);
  return 1;
}

int fly_world_set_state(const void *model_address, void *data_address,
                        int specification, const double *input, size_t length) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  const int required = model ? mj_stateSize(model, specification) : -1;
  if (!model || !data || required < 0 || length != (size_t)required ||
      (length && !input)) return 0;
  mj_setState(model, data, input, specification);
  return 1;
}

static int64_t num_length(const mjModel *model, int field) {
  if (!model) return -1;
  switch (field) {
    case FLY_WORLD_NUM_QPOS: return model->nq;
    case FLY_WORLD_NUM_QVEL: return model->nv;
    case FLY_WORLD_NUM_ACT: return model->na;
    case FLY_WORLD_NUM_QACC_WARMSTART: return model->nv;
    case FLY_WORLD_NUM_CTRL: return model->nu;
    case FLY_WORLD_NUM_QFRC_ACTUATOR: return model->nv;
    case FLY_WORLD_NUM_QFRC_CONSTRAINT: return model->nv;
    case FLY_WORLD_NUM_QFRC_APPLIED: return model->nv;
    case FLY_WORLD_NUM_XFRC_APPLIED: return model->nbody * 6;
    case FLY_WORLD_NUM_MOCAP_POS: return model->nmocap * 3;
    case FLY_WORLD_NUM_MOCAP_QUAT: return model->nmocap * 4;
    case FLY_WORLD_NUM_USERDATA: return model->nuserdata;
    case FLY_WORLD_NUM_XPOS: return model->nbody * 3;
    case FLY_WORLD_NUM_XQUAT: return model->nbody * 4;
    case FLY_WORLD_NUM_XMAT: return model->nbody * 9;
    case FLY_WORLD_NUM_GEOM_XPOS: return model->ngeom * 3;
    case FLY_WORLD_NUM_GEOM_XMAT: return model->ngeom * 9;
    case FLY_WORLD_NUM_SENSORDATA: return model->nsensordata;
    case FLY_WORLD_NUM_TIME: return 1;
    case FLY_WORLD_NUM_ACTUATOR_FORCE: return model->nout;
    case FLY_WORLD_NUM_XIPOS: return model->nbody * 3;
    case FLY_WORLD_NUM_GEOM_SIZE: return model->ngeom * 3;
    case FLY_WORLD_NUM_GEOM_POS: return model->ngeom * 3;
    case FLY_WORLD_NUM_GEOM_QUAT: return model->ngeom * 4;
    case FLY_WORLD_NUM_GEOM_RGBA: return model->ngeom * 4;
    case FLY_WORLD_NUM_GEOM_FRICTION: return model->ngeom * 3;
    case FLY_WORLD_NUM_ACTUATOR_FORCERANGE: return model->nactuator * 2;
    case FLY_WORLD_NUM_ACTUATOR_GAINPRM: return model->nactuator * mjNGAIN;
    case FLY_WORLD_NUM_MAT_RGBA: return model->nmat * 4;
    case FLY_WORLD_NUM_MESH_VERT: return model->nmeshvert * 3;
    case FLY_WORLD_NUM_MESH_NORMAL: return model->nmeshnormal * 3;
    default: return -1;
  }
}

int64_t fly_world_num_len(const void *model_address, int field) {
  return num_length((const mjModel *)model_address, field);
}

static int valid_range(int64_t length, size_t offset, size_t count) {
  return length >= 0 && offset <= (size_t)length &&
         count <= (size_t)length - offset;
}

#define READ_NUM_CASE(id, source) \
  case id: for (size_t i = 0; i < count; ++i) output[i] = (double)(source)[offset + i]; break

int fly_world_num_read(const void *model_address, const void *data_address,
                       int field, size_t offset, double *output,
                       size_t count) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  const int64_t length = num_length(model, field);
  if (!model || !valid_range(length, offset, count) || (count && !output) ||
      (field < 100 && !data)) return 0;
  switch (field) {
    READ_NUM_CASE(FLY_WORLD_NUM_QPOS, data->qpos);
    READ_NUM_CASE(FLY_WORLD_NUM_QVEL, data->qvel);
    READ_NUM_CASE(FLY_WORLD_NUM_ACT, data->act);
    READ_NUM_CASE(FLY_WORLD_NUM_QACC_WARMSTART, data->qacc_warmstart);
    READ_NUM_CASE(FLY_WORLD_NUM_CTRL, data->ctrl);
    READ_NUM_CASE(FLY_WORLD_NUM_QFRC_ACTUATOR, data->qfrc_actuator);
    READ_NUM_CASE(FLY_WORLD_NUM_QFRC_CONSTRAINT, data->qfrc_constraint);
    READ_NUM_CASE(FLY_WORLD_NUM_QFRC_APPLIED, data->qfrc_applied);
    READ_NUM_CASE(FLY_WORLD_NUM_XFRC_APPLIED, data->xfrc_applied);
    READ_NUM_CASE(FLY_WORLD_NUM_MOCAP_POS, data->mocap_pos);
    READ_NUM_CASE(FLY_WORLD_NUM_MOCAP_QUAT, data->mocap_quat);
    READ_NUM_CASE(FLY_WORLD_NUM_USERDATA, data->userdata);
    READ_NUM_CASE(FLY_WORLD_NUM_XPOS, data->xpos);
    READ_NUM_CASE(FLY_WORLD_NUM_XQUAT, data->xquat);
    READ_NUM_CASE(FLY_WORLD_NUM_XMAT, data->xmat);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_XPOS, data->geom_xpos);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_XMAT, data->geom_xmat);
    READ_NUM_CASE(FLY_WORLD_NUM_SENSORDATA, data->sensordata);
    READ_NUM_CASE(FLY_WORLD_NUM_ACTUATOR_FORCE, data->actuator_force);
    READ_NUM_CASE(FLY_WORLD_NUM_XIPOS, data->xipos);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_SIZE, model->geom_size);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_POS, model->geom_pos);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_QUAT, model->geom_quat);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_RGBA, model->geom_rgba);
    READ_NUM_CASE(FLY_WORLD_NUM_GEOM_FRICTION, model->geom_friction);
    READ_NUM_CASE(FLY_WORLD_NUM_ACTUATOR_FORCERANGE, model->actuator_forcerange);
    READ_NUM_CASE(FLY_WORLD_NUM_ACTUATOR_GAINPRM, model->actuator_gainprm);
    READ_NUM_CASE(FLY_WORLD_NUM_MAT_RGBA, model->mat_rgba);
    READ_NUM_CASE(FLY_WORLD_NUM_MESH_VERT, model->mesh_vert);
    READ_NUM_CASE(FLY_WORLD_NUM_MESH_NORMAL, model->mesh_normal);
    case FLY_WORLD_NUM_TIME:
      if (count) output[0] = data->time;
      break;
    default: return 0;
  }
  return 1;
}

#undef READ_NUM_CASE
#define WRITE_NUM_CASE(id, target) \
  case id: for (size_t i = 0; i < count; ++i) (target)[offset + i] = input[i]; break
#define WRITE_FLOAT_CASE(id, target) \
  case id: for (size_t i = 0; i < count; ++i) (target)[offset + i] = (float)input[i]; break

int fly_world_num_write(void *model_address, void *data_address, int field,
                        size_t offset, const double *input, size_t count) {
  mjModel *model = (mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  const int64_t length = num_length(model, field);
  if (!model || !valid_range(length, offset, count) || (count && !input) ||
      (field < 100 && !data)) return 0;
  switch (field) {
    WRITE_NUM_CASE(FLY_WORLD_NUM_QPOS, data->qpos);
    WRITE_NUM_CASE(FLY_WORLD_NUM_QVEL, data->qvel);
    WRITE_NUM_CASE(FLY_WORLD_NUM_ACT, data->act);
    WRITE_NUM_CASE(FLY_WORLD_NUM_QACC_WARMSTART, data->qacc_warmstart);
    WRITE_NUM_CASE(FLY_WORLD_NUM_CTRL, data->ctrl);
    WRITE_NUM_CASE(FLY_WORLD_NUM_QFRC_APPLIED, data->qfrc_applied);
    WRITE_NUM_CASE(FLY_WORLD_NUM_XFRC_APPLIED, data->xfrc_applied);
    WRITE_NUM_CASE(FLY_WORLD_NUM_MOCAP_POS, data->mocap_pos);
    WRITE_NUM_CASE(FLY_WORLD_NUM_MOCAP_QUAT, data->mocap_quat);
    WRITE_NUM_CASE(FLY_WORLD_NUM_USERDATA, data->userdata);
    WRITE_NUM_CASE(FLY_WORLD_NUM_GEOM_SIZE, model->geom_size);
    WRITE_NUM_CASE(FLY_WORLD_NUM_GEOM_POS, model->geom_pos);
    WRITE_NUM_CASE(FLY_WORLD_NUM_GEOM_QUAT, model->geom_quat);
    WRITE_FLOAT_CASE(FLY_WORLD_NUM_GEOM_RGBA, model->geom_rgba);
    WRITE_NUM_CASE(FLY_WORLD_NUM_GEOM_FRICTION, model->geom_friction);
    WRITE_NUM_CASE(FLY_WORLD_NUM_ACTUATOR_FORCERANGE, model->actuator_forcerange);
    WRITE_NUM_CASE(FLY_WORLD_NUM_ACTUATOR_GAINPRM, model->actuator_gainprm);
    WRITE_FLOAT_CASE(FLY_WORLD_NUM_MAT_RGBA, model->mat_rgba);
    case FLY_WORLD_NUM_TIME:
      if (count) data->time = input[0];
      break;
    default: return 0;  // derived quantities are read-only
  }
  return 1;
}

#undef WRITE_NUM_CASE
#undef WRITE_FLOAT_CASE

static int64_t int_length(const mjModel *model, int field) {
  if (!model) return -1;
  switch (field) {
    case FLY_WORLD_INT_GEOM_BODYID:
    case FLY_WORLD_INT_GEOM_TYPE:
    case FLY_WORLD_INT_GEOM_MATID:
    case FLY_WORLD_INT_GEOM_CONTYPE:
    case FLY_WORLD_INT_GEOM_CONAFFINITY:
    case FLY_WORLD_INT_GEOM_GROUP:
    case FLY_WORLD_INT_GEOM_DATAID: return model->ngeom;
    case FLY_WORLD_INT_MESH_VERTADR:
    case FLY_WORLD_INT_MESH_VERTNUM:
    case FLY_WORLD_INT_MESH_NORMALADR:
    case FLY_WORLD_INT_MESH_NORMALNUM:
    case FLY_WORLD_INT_MESH_FACEADR:
    case FLY_WORLD_INT_MESH_FACENUM: return model->nmesh;
    case FLY_WORLD_INT_MESH_FACE: return model->nmeshface * 3;
    default: return -1;
  }
}

int64_t fly_world_int_len(const void *model_address, int field) {
  return int_length((const mjModel *)model_address, field);
}

#define READ_INT_CASE(id, source) \
  case id: for (size_t i = 0; i < count; ++i) output[i] = (int32_t)(source)[offset + i]; break

int fly_world_int_read(const void *model_address, int field, size_t offset,
                       int32_t *output, size_t count) {
  const mjModel *model = (const mjModel *)model_address;
  const int64_t length = int_length(model, field);
  if (!model || !valid_range(length, offset, count) || (count && !output)) return 0;
  switch (field) {
    READ_INT_CASE(FLY_WORLD_INT_GEOM_BODYID, model->geom_bodyid);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_TYPE, model->geom_type);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_MATID, model->geom_matid);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_DATAID, model->geom_dataid);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_CONTYPE, model->geom_contype);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_CONAFFINITY, model->geom_conaffinity);
    READ_INT_CASE(FLY_WORLD_INT_GEOM_GROUP, model->geom_group);
    READ_INT_CASE(FLY_WORLD_INT_MESH_VERTADR, model->mesh_vertadr);
    READ_INT_CASE(FLY_WORLD_INT_MESH_VERTNUM, model->mesh_vertnum);
    READ_INT_CASE(FLY_WORLD_INT_MESH_NORMALADR, model->mesh_normaladr);
    READ_INT_CASE(FLY_WORLD_INT_MESH_NORMALNUM, model->mesh_normalnum);
    READ_INT_CASE(FLY_WORLD_INT_MESH_FACEADR, model->mesh_faceadr);
    READ_INT_CASE(FLY_WORLD_INT_MESH_FACENUM, model->mesh_facenum);
    READ_INT_CASE(FLY_WORLD_INT_MESH_FACE, model->mesh_face);
    default: return 0;
  }
  return 1;
}

#undef READ_INT_CASE
#define WRITE_INT_CASE(id, target) \
  case id: for (size_t i = 0; i < count; ++i) (target)[offset + i] = (int)input[i]; break

int fly_world_int_write(void *model_address, int field, size_t offset,
                        const int32_t *input, size_t count) {
  mjModel *model = (mjModel *)model_address;
  const int64_t length = int_length(model, field);
  if (!model || !valid_range(length, offset, count) || (count && !input)) return 0;
  switch (field) {
    WRITE_INT_CASE(FLY_WORLD_INT_GEOM_CONTYPE, model->geom_contype);
    WRITE_INT_CASE(FLY_WORLD_INT_GEOM_CONAFFINITY, model->geom_conaffinity);
    WRITE_INT_CASE(FLY_WORLD_INT_GEOM_GROUP, model->geom_group);
    default: return 0;
  }
  return 1;
}

#undef WRITE_INT_CASE

int fly_world_xbody_velocity_all(const void *model_address,
                                 const void *data_address, double *output,
                                 size_t length) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !data || length != (size_t)model->nbody * 6 ||
      (length && !output)) return 0;
  for (int body = 0; body < model->nbody; ++body) {
    mj_objectVelocity(model, data, mjOBJ_XBODY, body, output + 6 * body, 0);
  }
  return 1;
}

int fly_world_xbody_velocity_selected(const void *model_address,
                                      const void *data_address,
                                      const int32_t *bodies, size_t count,
                                      double *output) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !data || (count && (!bodies || !output))) return 0;
  for (size_t row = 0; row < count; ++row) {
    if (bodies[row] < 0 || bodies[row] >= model->nbody) return 0;
    mj_objectVelocity(model, data, mjOBJ_XBODY, bodies[row], output + row * 6, 0);
  }
  return 1;
}

int fly_world_contact_count(const void *data_address) {
  const mjData *data = (const mjData *)data_address;
  return data ? data->ncon : -1;
}

int fly_world_contact_records(const void *model_address,
                              const void *data_address, double *records,
                              double *distances, size_t capacity_records) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !data || capacity_records < (size_t)data->ncon ||
      (data->ncon && !records)) return -1;
  for (int index = 0; index < data->ncon; ++index) {
    const mjContact *contact = data->contact + index;
    double *record = records + (size_t)index * FLY_WORLD_CONTACT_STRIDE;
    double force[6];
    mj_contactForce(model, data, index, force);
    record[0] = contact->geom[0];
    record[1] = contact->geom[1];
    for (int axis = 0; axis < 3; ++axis) record[2 + axis] = contact->pos[axis];
    for (int cell = 0; cell < 9; ++cell) record[5 + cell] = contact->frame[cell];
    for (int axis = 0; axis < 6; ++axis) record[14 + axis] = force[axis];
    if (distances) distances[index] = contact->dist;
  }
  return data->ncon;
}

int fly_world_ray(const void *model_address, const void *data_address,
                  const double origin[3], const double direction[3],
                  const uint8_t geom_group[6], int include_static,
                  int excluded_body, int *geom, double *distance,
                  double normal[3]) {
  const mjModel *model = (const mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !data || !origin || !direction || !geom || !distance ||
      excluded_body < -1 || excluded_body >= model->nbody) return 0;
  *geom = -1;
  *distance = mj_ray(model, data, origin, direction, geom_group,
                     include_static != 0, excluded_body, geom, normal);
  return 1;
}

int fly_world_multi_ray(const void *model_address, void *data_address,
                        const double origin[3], const double *directions,
                        const uint8_t geom_group[6], int include_static,
                        int excluded_body, int *geoms, double *distances,
                        double *normals, int ray_count, double cutoff) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || !origin || ray_count < 0 ||
      (ray_count && (!directions || !geoms || !distances)) ||
      excluded_body < -1 || excluded_body >= model->nbody ||
      !isfinite(cutoff) || cutoff < 0) return 0;
  mj_multiRay(model, data, origin, directions, geom_group, include_static != 0,
              excluded_body, geoms, distances, normals, ray_count, cutoff);
  return 1;
}

static int body_is_hidden(int body, const int32_t *hidden, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (hidden[index] == body) return 1;
  }
  return 0;
}

int fly_world_multi_ray_masked(void *model_address, void *data_address,
                               const double origin[3],
                               const double *directions, int ray_count,
                               const int32_t *hidden_bodies,
                               size_t hidden_body_count, double cutoff,
                               int *geoms, double *distances,
                               double *normals) {
  mjModel *model = (mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || !origin || ray_count < 0 ||
      (ray_count && (!directions || !geoms || !distances)) ||
      (hidden_body_count && !hidden_bodies) || !isfinite(cutoff) || cutoff < 0)
    return 0;
  for (size_t index = 0; index < hidden_body_count; ++index) {
    if (hidden_bodies[index] < 0 || hidden_bodies[index] >= model->nbody)
      return 0;
  }

  int *saved_groups = NULL;
  if (model->ngeom) {
    saved_groups = (int *)malloc((size_t)model->ngeom * sizeof(*saved_groups));
    if (!saved_groups) return 0;
  }
  for (int geom = 0; geom < model->ngeom; ++geom) {
    saved_groups[geom] = model->geom_group[geom];
    if (body_is_hidden(model->geom_bodyid[geom], hidden_bodies,
                       hidden_body_count)) {
      model->geom_group[geom] = 5;
    } else if (model->geom_group[geom] == 5) {
      model->geom_group[geom] = 0;
    }
  }
  const uint8_t group_mask[6] = {1, 1, 1, 1, 1, 0};
  mj_multiRay(model, data, origin, directions, group_mask, 1, -1, geoms,
              distances, normals, ray_count, cutoff);
  for (int geom = 0; geom < model->ngeom; ++geom)
    model->geom_group[geom] = saved_groups[geom];
  free(saved_groups);
  return 1;
}

int fly_world_rays(void *model_address, const void *data_address,
                   const double *origins, const double *directions,
                   int ray_count, double cutoff, int *geoms, double *distances) {
  mjModel *model = (mjModel *)model_address;
  const mjData *data = (const mjData *)data_address;
  if (!model || !data || !origins || !directions || ray_count < 0 ||
      !isfinite(cutoff) || cutoff <= 0 || (ray_count && (!geoms || !distances))) return 0;
  int *saved_groups = model->ngeom ? (int *)malloc((size_t)model->ngeom * sizeof(int)) : NULL;
  if (model->ngeom && !saved_groups) return 0;
  for (int geom = 0; geom < model->ngeom; ++geom) {
    saved_groups[geom] = model->geom_group[geom];
    model->geom_group[geom] = (model->geom_contype[geom] || model->geom_conaffinity[geom]) ? 0 : 5;
  }
  const uint8_t collision_solids[6] = {1, 0, 0, 0, 0, 0};
  for (int i = 0; i < ray_count; ++i) {
    int geom = -1;
    const mjtNum d = mj_ray(model, data, origins + i * 3, directions + i * 3,
                            collision_solids, 1, -1, &geom, NULL);
    geoms[i] = d >= 0 && d <= cutoff ? geom : -1;
    distances[i] = d >= 0 ? fmin((double)d, cutoff) : cutoff;
  }
  for (int geom = 0; geom < model->ngeom; ++geom) model->geom_group[geom] = saved_groups[geom];
  free(saved_groups); return 1;
}

int fly_world_endpoint_inside(void *model_address, void *data_address,
                              int proxy_geom, int source_geom_count,
                              const double *points, size_t point_count,
                              unsigned char *inside) {
  mjModel *model = (mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || proxy_geom < source_geom_count || proxy_geom >= model->ngeom ||
      source_geom_count < 0 || source_geom_count > model->ngeom ||
      (point_count && (!points || !inside))) return 0;
  mjtNum old_position[3]; mju_copy3(old_position, data->geom_xpos + proxy_geom * 3);
  for (size_t point = 0; point < point_count; ++point) {
    mju_copy3(data->geom_xpos + proxy_geom * 3, points + point * 3); inside[point] = 0;
    for (int geom = 0; geom < source_geom_count; ++geom) {
      if (!model->geom_contype[geom] && !model->geom_conaffinity[geom]) continue;
      if (model->geom_type[geom] != mjGEOM_PLANE && model->geom_type[geom] != mjGEOM_HFIELD) {
        const mjtNum *center = data->geom_xpos + geom * 3;
        const mjtNum dx = center[0] - points[point * 3];
        const mjtNum dy = center[1] - points[point * 3 + 1];
        const mjtNum dz = center[2] - points[point * 3 + 2];
        const mjtNum bound = model->geom_rbound[geom] + model->geom_rbound[proxy_geom];
        if (dx * dx + dy * dy + dz * dz > bound * bound) continue;
      }
      mjtNum from_to[6];
      if (mj_geomDistance(model, data, proxy_geom, geom, 1, from_to) <= 0) { inside[point] = 1; break; }
    }
  }
  mju_copy3(data->geom_xpos + proxy_geom * 3, old_position); return 1;
}

int fly_world_geom_distance(const void *model_address, void *data_address,
                            int geom1, int geom2, double maximum_distance,
                            double from_to[6], double *distance) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || !distance || geom1 < 0 || geom1 >= model->ngeom ||
      geom2 < 0 || geom2 >= model->ngeom || !isfinite(maximum_distance))
    return 0;
  *distance = mj_geomDistance(model, data, geom1, geom2, maximum_distance,
                              from_to);
  return 1;
}

int fly_world_body_name_to_id(const void *model_address, const char *name) {
  const mjModel *model = (const mjModel *)model_address;
  return model && name ? mj_name2id(model, mjOBJ_BODY, name) : -1;
}

int fly_world_geom_name_to_id(const void *model_address, const char *name) {
  const mjModel *model = (const mjModel *)model_address;
  return model && name ? mj_name2id(model, mjOBJ_GEOM, name) : -1;
}

size_t fly_world_geom_id_to_name(const void *model_address, int geom,
                                 char *output, size_t capacity) {
  const mjModel *model = (const mjModel *)model_address;
  if (!model || geom < 0 || geom >= model->ngeom) return 0;
  const char *name = mj_id2name(model, mjOBJ_GEOM, geom);
  if (!name) return 0;
  const size_t required = strlen(name) + 1;
  if (output && capacity) {
    const size_t copied = required < capacity ? required : capacity;
    memcpy(output, name, copied - 1);
    output[copied - 1] = '\0';
  }
  return required;
}
