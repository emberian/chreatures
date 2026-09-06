#include <math.h>
#include <mujoco/mujoco.h>

int chreatures_mujoco_header_version(void) { return mjVERSION_HEADER; }
int chreatures_mujoco_runtime_version(void) { return mj_version(); }

int chreatures_contact_batch(
    const void *model_address, void *data_address, int capacity,
    double timestep, double impulse_limit, double work_limit,
    int *geom1, int *geom2, double *positions, double *normals,
    double *relative_speed, double *impulse, double *impact_work,
    double *contact_force_norm, int geom_count, const int *geom_resident,
    const int *geom_entity, const int *resident_body, const double *resident_z,
    int resident_count, int *participant_resident, int *participant_entity,
    signed char *participant_side, double *participant_normal) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || capacity < data->ncon || resident_count < 0 ||
      geom_count != model->ngeom) return -1;
  for (int index = 0; index < data->ncon; ++index) {
    const mjContact *contact = data->contact + index;
    if (contact->geom1 < 0 || contact->geom1 >= model->ngeom ||
        contact->geom2 < 0 || contact->geom2 >= model->ngeom) {
      return -3;
    }
    geom1[index] = contact->geom1;
    geom2[index] = contact->geom2;
    for (int axis = 0; axis < 3; ++axis) {
      positions[index * 3 + axis] = contact->pos[axis];
      normals[index * 3 + axis] = contact->frame[axis];
    }
    double point_velocity[2][3];
    const int geoms[2] = {contact->geom1, contact->geom2};
    for (int side = 0; side < 2; ++side) {
      const int body = model->geom_bodyid[geoms[side]];
      double velocity[6];
      mj_objectVelocity(model, data, mjOBJ_BODY, body, velocity, 0);
      const double *origin = data->xpos + 3 * body;
      const double rx = contact->pos[0] - origin[0];
      const double ry = contact->pos[1] - origin[1];
      const double rz = contact->pos[2] - origin[2];
      point_velocity[side][0] = velocity[3] + velocity[1] * rz - velocity[2] * ry;
      point_velocity[side][1] = velocity[4] + velocity[2] * rx - velocity[0] * rz;
      point_velocity[side][2] = velocity[5] + velocity[0] * ry - velocity[1] * rx;
    }
    const double dx = point_velocity[1][0] - point_velocity[0][0];
    const double dy = point_velocity[1][1] - point_velocity[0][1];
    const double dz = point_velocity[1][2] - point_velocity[0][2];
    relative_speed[index] = fabs(dx * contact->frame[0] + dy * contact->frame[1] + dz * contact->frame[2]);
    double force[6];
    mj_contactForce(model, data, index, force);
    contact_force_norm[index] = sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
    impulse[index] = fmin(impulse_limit, fabs(force[0]) * timestep);
    impact_work[index] = fmin(work_limit, 0.5 * impulse[index] * relative_speed[index]);
    for (int side = 0; side < 2; ++side) {
      const int slot = 2 * index + side;
      const int resident = geom_resident[geoms[side]];
      const int other = geoms[1 - side];
      participant_resident[slot] = resident;
      participant_entity[slot] = geom_entity[other];
      participant_side[slot] = -1;
      for (int axis = 0; axis < 3; ++axis) participant_normal[3 * slot + axis] = 0.0;
      if (resident < 0) continue;
      if (resident >= resident_count || resident_body[resident] < 0 ||
          resident_body[resident] >= model->nbody) return -2;
      const double sign = side == 0 ? 1.0 : -1.0;
      const double nx = sign * contact->frame[0];
      const double ny = sign * contact->frame[1];
      const double nz = sign * contact->frame[2];
      if (fabs(nz) > 0.72 && contact->pos[2] < resident_z[resident]) continue;
      const int body = resident_body[resident];
      const double *rotation = data->xmat + 9 * body;
      const double dxp = contact->pos[0] - data->xpos[3 * body];
      const double dyp = contact->pos[1] - data->xpos[3 * body + 1];
      const double dzp = contact->pos[2] - data->xpos[3 * body + 2];
      participant_side[slot] =
          dxp * rotation[1] + dyp * rotation[4] + dzp * rotation[7] >= 0.0;
      participant_normal[3 * slot] =
          rotation[0] * nx + rotation[3] * ny + rotation[6] * nz;
      participant_normal[3 * slot + 1] =
          rotation[1] * nx + rotation[4] * ny + rotation[7] * nz;
      participant_normal[3 * slot + 2] =
          rotation[2] * nx + rotation[5] * ny + rotation[8] * nz;
    }
  }
  return data->ncon;
}
