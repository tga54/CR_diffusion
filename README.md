### main ###
  We calculate the diffusion of cosmic ray in the galaxy with absorption boundary condition. Due to the symmetry of the system, we ignored the angular coordinate and set reflective boundaries in Z=0 and R=0. The cosmic rays are injected from a disk and thus we use numerical methods to solve the diffuion equation in cylinderical coordinate system. We use openmp to calculate spatial grid values in parallezation. 

Function initialize_to_disk, initialize_to_0 can set the injection of primary or secondary CRs. 
Function initialize_H can set the gas distribution to flat distribution or spatial-dependent distribution.
