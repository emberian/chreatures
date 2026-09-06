#include <math.h>
#include <mujoco/mujoco.h>

int chreatures_acoustic_hinges(const void *ma, void *da, int n,
 const int *dofs, const double *damping, const double *limit, double dt, double *work) {
 const mjModel *m=(const mjModel*)ma; mjData *d=(mjData*)da;
 if(!m||!d||n<0) return -1;
 for(int i=0;i<n;i++) if(dofs[i]<-1||dofs[i]>=m->nv) return -1;
 for(int i=0;i<n;i++){ work[i]=0; if(dofs[i]<0) continue; double v=d->qvel[dofs[i]];
  double t=fmax(-limit[i],fmin(limit[i],-damping[i]*v)); d->qfrc_applied[dofs[i]]+=t;
  work[i]=fmax(0.0,-t*v*dt); }
 return n;
}

int chreatures_acoustic_sample(const void *ma, void *da, int n, const int *bodies,
 const double *offsets, const double *energy, const double *gain, const double *reference,
 const double *range, const double *occlusion, const double *listener, int exclude, double *out) {
 const mjModel *m=(const mjModel*)ma; mjData *d=(mjData*)da; if(!m||!d||n<0||exclude<-1||exclude>=m->nbody) return -1;
 for(int i=0;i<n;i++) if(bodies[i]<0||bodies[i]>=m->nbody) return -1;
 out[0]=out[1]=out[2]=0;
 for(int i=0;i<n;i++){
  const int b=bodies[i]; const double *r=d->xmat+9*b; double source[3];
  for(int a=0;a<3;a++) source[a]=d->xpos[3*b+a]+r[3*a]*offsets[3*i]+r[3*a+1]*offsets[3*i+1]+r[3*a+2]*offsets[3*i+2];
  double delta[3]={source[0]-listener[0],source[1]-listener[1],source[2]-listener[2]};
  double dist=sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]), vis=1;
  if(dist>1e-8){ double dir[3]={delta[0]/dist,delta[1]/dist,delta[2]/dist}, normal[3]; int geom=-1;
   double hit=mj_ray(m,d,listener,dir,NULL,1,exclude,&geom,normal);
   int source_body=geom>=0?m->geom_bodyid[geom]:-1;
   if(!(hit<0||hit>=dist-0.07||source_body==b)) vis=occlusion[i]; }
  double atten=vis/(1+(dist/range[i])*(dist/range[i]));
  for(int t=0;t<3;t++) out[t]+=atten*gain[i]*sqrt(energy[3*i+t]/reference[i]);
 }
 for(int t=0;t<3;t++) out[t]=fmax(0,fmin(2,out[t])); return n;
}
