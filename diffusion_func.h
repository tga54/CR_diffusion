#ifndef FUNC_H
#define FUNC_H

#include <stdio.h>

#define NR 721
#define NZ 331
#define NT 100000
extern const int NR1, NZ1;
extern const double dR1, dZ1, R_inj, Z_inj;
extern const double sigma_pC, sigma_pB, sigma_pC2B, c, yr2s, ratio;
extern const double ratio_threshold;
extern double R[NR], Z[NZ], dR[NR], dZ[NZ];

void initialize_grids(void);
void initialize_to_disk(double *ndis, const double dT);
void initialize_to_0(double *ndis);
void initialize_from_file(const char *filename, double *ndis);
void initialize_H(const double nH);
void apply_boundary_conditions(double *ndis);
void solve_diffusion_equation(double *ndis_C, double *ndis_B, double *NC, double *NB, const double D, const double dT);
void write_array2D_to_txt(const char *filename, const double *ndis);
void write_array1D_to_txt(const char *filename, const double *N);
#endif
