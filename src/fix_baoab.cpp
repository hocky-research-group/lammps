/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "stdio.h"
#include "string.h"
#include "fix_baoab.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "mpi.h"
#include "math.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "math_extra.h"
#include "compute.h"
#include "domain.h"
#include "group.h"
#include "universe.h"
#include <time.h>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBAOAB::FixBAOAB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

// fix 1 all baoab  gamma  temp  seed 

    
  if (narg < 6) error->all(FLERR,"Illegal fix baoab command :: too few args");
  
   
  gamma = force->numeric(FLERR,arg[3]);
  t_target = force->numeric(FLERR,arg[4]); 
  seed = force->inumeric(FLERR,arg[5]);
  

  if (t_target < 0.0) error->all(FLERR,"Target temperature must be >= 0 ");
  if (gamma < 0.0) error->all(FLERR,"Friction must be >= 0 ");
  if (seed < 0) error->all(FLERR,"Seed must be an integer >= 0 "); 
  
  if (seed==0) {
  // Let's use a random seed!
    seed = ( int)( 1 + (time(NULL) % 10000000)); 
  }
  
  seed = seed + universe->iworld;
  
  tsqrt = sqrt( t_target);  
  random = new RanMars(lmp,seed);
  
  if (screen) {
   fprintf(screen, ">Running Langevin dynamics at seed %d and temperature %.1f K with friction %f ps^-1\n",
	   seed , t_target, gamma );
  }
    
  if (logfile) {
   fprintf(logfile, ">Running Langevin dynamics at seed %d and temperature %.1f K with friction %f ps^-1\n",
	   seed , t_target, gamma );
  }
//   if (universe->iworld==0) {
//     
//     if (universe->uscreen)
//       fprintf(universe->uscreen, " >>\n >> comrep setup ok\n >> Run for %d steps then communicate between %d replicas with %d atoms at temperature %.1f  (seed %d+i, boost %.2f).\n >>\n", 
// 	      comrepbegin, universe->nprocs, atom->natoms, t_target, seed, LAMBDA+1);
//       
//       
//     if (universe->ulogfile)
//       fprintf(universe->ulogfile, " >>\n >> comrep setup ok\n >> Run for %d steps then communicate between %d replicas with %d atoms at temperature %.1f  (seed %d+i, boost %.2f).\n >>\n", 
// 	      comrepbegin, universe->nprocs, atom->natoms, t_target, seed,LAMBDA+1);
//  
//   }
   
  sig = 0;
  
}

FixBAOAB::~FixBAOAB()
{
  delete random; 
  
}

/* ---------------------------------------------------------------------- */

int FixBAOAB::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBAOAB::init()
{

  double boltz = force->boltz;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  
  dtv = 0.5*update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  
  

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
  
    sig =  tsqrt * sqrt(boltz/mvv2e);
  

  c1 = exp(-update->dt * gamma*0.001);
  c3 = sig * sqrt(1-c1*c1);
  
  
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixBAOAB::initial_integrate(int vflag)
{
  double dtfm;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double sqrm;
  int tdim;
  tagint *tag = atom->tag;
  double R[ universe->nprocs ] ;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double ftm2v = force->ftm2v;
  double tv, stats[5];
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int ii,jj;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
 
  
  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	sqrm = 1.0 / sqrt( rmass[i] ); 
        dtfm = dtf / rmass[i];
	// B
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	// O
	  v[i][0] = c1 * v[i][0] + c3 * sqrm * random->gaussian(); 
	  v[i][1] = c1 * v[i][1] + c3 * sqrm * random->gaussian(); 
	  v[i][2] = c1 * v[i][2] + c3 * sqrm * random->gaussian(); 
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	sqrm = 1.0 / sqrt( mass[type[i]] ); 
        dtfm = dtf / mass[type[i]];
	// B
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	// O
	  v[i][0] = c1 * v[i][0] + c3 * sqrm * random->gaussian(); 
	  v[i][1] = c1 * v[i][1] + c3 * sqrm * random->gaussian(); 
	  v[i][2] = c1 * v[i][2] + c3 * sqrm * random->gaussian(); 
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
      }
  }
  
  
   
  
}

/* ---------------------------------------------------------------------- */

void FixBAOAB::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBAOAB::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = 0.5*step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBAOAB::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBAOAB::reset_dt()
{
  dtv = 0.5*update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
