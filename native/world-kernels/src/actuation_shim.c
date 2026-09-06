#include <math.h>
#include <mujoco/mujoco.h>

enum { C_FREQUENCY, C_STANCE, C_HIP_AMPLITUDE, C_KNEE_STANCE,
       C_KNEE_SWING, C_IDLE_KNEE, C_TORQUE_LIMIT, C_TURN_GAIN,
       C_HIP_KP, C_KNEE_KP, C_HIP_KD, C_KNEE_KD, C_POSTURE_KP,
       C_POSTURE_KD, C_POSTURE_LIMIT, C_COUNT };
enum { D_FORWARD, D_TURN, D_ENERGY, D_FATIGUE, D_SCALE, D_COUNT };

int chreatures_actuation_batch(
    const void *model_address, void *data_address, int resident_count,
    int leg_count, const int *root_body, const int *qpos_address,
    const int *dof_address, const double *side, const double *phase,
    const double *controller, const double *dynamic, const int *grip_body,
    double world_time, double timestep, int phase_kind, double *work) {
  const mjModel *model = (const mjModel *)model_address;
  mjData *data = (mjData *)data_address;
  if (!model || !data || resident_count < 0 || leg_count <= 0) return -1;
  for (int resident = 0; resident < resident_count; ++resident) {
    const double *c = controller + resident * C_COUNT;
    const double *d = dynamic + resident * D_COUNT;
    const double forward = d[D_FORWARD], turn = d[D_TURN];
    const double scale = d[D_SCALE];
    const int root = root_body[resident];
    const double *rotation = data->xmat + 9 * root;
    double velocity[6];
    mj_objectVelocity(model, data, mjOBJ_BODY, root, velocity, 0);
    if (phase_kind == 0) {
    const double activity = fmax(fabs(forward), fabs(turn));
    const double strength = (1.0 - 0.72 * d[D_FATIGUE]) *
                            (0.18 + 0.82 * d[D_ENERGY]);
    const double frequency = c[C_FREQUENCY] * (0.32 + 0.68 * activity);
    const double torque_limit = c[C_TORQUE_LIMIT] * strength;
    double gait_power = 0.0;
    for (int leg = 0; leg < leg_count; ++leg) {
      const int item = resident * leg_count + leg;
      double drive = forward + side[item] * c[C_TURN_GAIN] * turn;
      drive = fmax(-1.0, fmin(1.0, drive));
      double hip_target, knee_target;
      if (activity < 1e-4 || fabs(drive) < 1e-4) {
        hip_target = 0.0;
        knee_target = side[item] * c[C_IDLE_KNEE];
      } else {
        double cycle = fmod(world_time * frequency + phase[item], 1.0);
        const double stride = 0.30 + 0.70 * fabs(drive);
        double sweep;
        if (cycle < c[C_STANCE]) {
          sweep = 1.0 - 2.0 * cycle / c[C_STANCE];
          knee_target = side[item] * c[C_KNEE_STANCE];
        } else {
          sweep = -1.0 + 2.0 * (cycle - c[C_STANCE]) / (1.0 - c[C_STANCE]);
          knee_target = side[item] * c[C_KNEE_SWING];
        }
        hip_target = -side[item] * copysign(1.0, drive) *
                     c[C_HIP_AMPLITUDE] * stride * sweep;
      }
      const double targets[2] = {hip_target, knee_target};
      for (int kind = 0; kind < 2; ++kind) {
        const int joint = item * 2 + kind;
        const int qadr = qpos_address[joint], dadr = dof_address[joint];
        const double kp = kind == 0 ? c[C_HIP_KP] : c[C_KNEE_KP];
        const double kd = kind == 0 ? c[C_HIP_KD] : c[C_KNEE_KD];
        double torque = kp * (targets[kind] - data->qpos[qadr]) - kd * data->qvel[dadr];
        torque = fmax(-torque_limit, fmin(torque_limit, torque)) * scale;
        data->qfrc_applied[dadr] = torque;
        gait_power += torque * data->qvel[dadr];
      }
    }
    double correction[3] = {
      rotation[5] * c[C_POSTURE_KP] - velocity[0] * c[C_POSTURE_KD],
      -rotation[2] * c[C_POSTURE_KP] - velocity[1] * c[C_POSTURE_KD], 0.0
    };
    const double norm = hypot(correction[0], correction[1]);
    if (norm > c[C_POSTURE_LIMIT]) {
      correction[0] *= c[C_POSTURE_LIMIT] / norm;
      correction[1] *= c[C_POSTURE_LIMIT] / norm;
    }
    for (int axis = 0; axis < 2; ++axis) {
      correction[axis] *= scale;
      data->xfrc_applied[6 * root + 3 + axis] += correction[axis];
      gait_power += correction[axis] * velocity[axis];
    }
    work[resident] += fmax(0.0, gait_power) * timestep;
    }
    if (phase_kind != 1) continue;
    const int object = grip_body[resident];
    if (object >= 0) {
      double object_velocity[6];
      mj_objectVelocity(model, data, mjOBJ_BODY, object, object_velocity, 0);
      double force[3];
      double norm2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double target = data->xpos[3 * root + axis] +
          rotation[axis * 3] * 0.17 + (axis == 2 ? 0.04 : 0.0);
        force[axis] = ((target - data->xpos[3 * object + axis]) * 15.0 -
          (object_velocity[3 + axis] - velocity[3 + axis]) * 1.1) * scale;
        norm2 += force[axis] * force[axis];
      }
      const double force_norm = sqrt(norm2);
      if (force_norm > 8.0) for (int axis = 0; axis < 3; ++axis) force[axis] *= 8.0 / force_norm;
      double grip_power = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        data->xfrc_applied[6 * object + axis] += force[axis];
        data->xfrc_applied[6 * root + axis] -= force[axis];
        grip_power += force[axis] * object_velocity[3 + axis] - force[axis] * velocity[3 + axis];
      }
      work[resident] += fmax(0.0, grip_power) * timestep;
    }
  }
  return resident_count;
}
