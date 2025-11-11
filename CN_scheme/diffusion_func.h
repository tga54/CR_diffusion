#ifndef FUNC_H
#define FUNC_H

#include <stdio.h>

#define NR 201
#define NZ 201
#define NT 10000
extern const int NR1, NZ1;
extern const double dR1, dZ1, dR2, dZ2, R_disk, Z_disk, R_inj;
extern const double sigma_pC, sigma_pB, sigma_pC2B, c, yr2s, ratio;
extern const double ratio_threshold;
extern double R[NR], Z[NZ], dR[NR], dZ[NZ];

typedef struct {
    int NR_BH, NZ_BH;
    double dR_BH, dZ_BH, dT_BH, Dd_BH, Dh_BH, ss_time_BH, ndis_H_BH, R_disk_BH;
    char description_BH[128];
} BinHeader;


void initialize_grids_linear(void);
void initialize_grids_log(void);
void initialize_to_disk(double *ndis, const double dT);
void initialize_to_src(double *ndis, const double dT);
void initialize_to_0(double *ndis);
void initialize_from_file(const char *filename, double *ndis);
void initialize_H(double *ndis_H, const double nH);
void initialize_D(double *D, const double Dd, const double Dh);
void apply_boundary_conditions(double *ndis);
void solve_diffusion_equation(double *ndis_C, double *ndis_B, double *ndis_H, double *NC, double *NB, double *D, const double dT, double *ss_time);
void thomas_solve(const double *a, const double *b, const double *c, const double *d, double *x, int n);
void CN_scheme(double *ndis, double *src, double *ndis_H, double *dT_tau, double *temp, double sigma, double D, double dT);
void solve_diffusion_equation_CN(double *ndis_C, double *ndis_B, double *ndis_H, double *NC, double *NB, double D, const double dT, double *ss_time);
void write_array2D_to_txt(const char *filename, const double *ndis);
void write_array1D_to_txt(const char *filename, const double *N);
void write_array_to_bin(const char *filename, const double *arr, const int size);
void write_array_header_to_bin(const char *filename, const double *arr, const int size, const double dT, const double *ss_time, const double Dd, const double Dh, const double gas_type, const char *grid_type);

#endif
