#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "diffusion_func.h"

int main(void) {

    // Set number of threads to the number of CPUs
    int ncore = omp_get_num_procs(); 
    omp_set_num_threads(ncore);

    // Dynamically allocate memory for ndis and temp
    double *ndis_C = malloc(NR * NZ * sizeof(double));
    double *ndis_B = malloc(NR * NZ * sizeof(double));
    double *ndis_H = malloc(NR * NZ * sizeof(double));
    double *D = malloc(NR * NZ * sizeof(double));

    double *NC = (double *)malloc(NT * sizeof(double));  // typically size of NT, set 1 to avoid large memory allocation when unrequired
    double *NB = (double *)malloc(NT * sizeof(double));
    double Rg[8] = {1., 10., 100., 1000., 10000., 100000., 1000000.}; // Rigidity

    double dT, Dh, Dd, ss_time = 0.0, gas_type = -1.0;
    char grid_type[10] = "log";
    initialize_grids_log();
    initialize_H(ndis_H, gas_type);

    for (int e = 0; e < 1; e++) {

        Dd = 3.3e-8*pow(Rg[e], 1./3.); // D = 1e28 (R/GV)^(1/3) cm^2/s = 3.3e-8 (R/GV)^(1/3) kpc^2/yr
        Dh = 1.0 * Dd;
        dT = dZ1 * dZ1 / Dd / 1.0; // dT <= (dR^2+dZ^2) / D
        // Initialize particle density and apply boundary conditions
        initialize_to_disk(ndis_C, dT);
        initialize_to_0(ndis_B);
        initialize_D(D, Dd, Dh);
        // solve_diffusion_equation(ndis_C, ndis_B, ndis_H, NC, NB, D, dT, &ss_time);
        solve_diffusion_equation_CN(ndis_C, ndis_B, ndis_H, NC, NB, Dd, dT, &ss_time);

        // output as bin file
        char fn1[100], fn2[100], fn3[100], fn4[100], name_body[100];
        snprintf(name_body, sizeof(name_body), "%dGV_R%d_Z%d_inj%d_CN_realgas_T10000_dt10.bin", (int)Rg[e], (int)R[NR-1], (int)Z[NZ-1], (int)R_inj );
        snprintf(fn1, sizeof(fn1), "ndis_C_%s", name_body);
        snprintf(fn2, sizeof(fn2), "ndis_B_%s", name_body);
        snprintf(fn3, sizeof(fn3), "NC_%s", name_body);
        snprintf(fn4, sizeof(fn4), "NB_%s", name_body);
        
        write_array_header_to_bin(fn1, ndis_C, NR * NZ, dT, &ss_time, Dd, Dh, gas_type, grid_type);
        write_array_header_to_bin(fn2, ndis_B, NR * NZ, dT, &ss_time, Dd, Dh, gas_type, grid_type);
    }

    // Free the dynamically allocated memory
    free(ndis_C);
    free(ndis_B);
    free(ndis_H);
    free(NC);
    free(NB);
    free(D);
    return 0;
}



