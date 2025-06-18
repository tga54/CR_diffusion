#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "diffusion_func.h"

int main(void) {

    // Set number of threads to 10
    omp_set_num_threads(10);

    // Dynamically allocate memory for ndis and temp
    double *ndis_C = malloc(NR * NZ * sizeof(double));
    double *ndis_B = malloc(NR * NZ * sizeof(double));
    double *NC = (double *)malloc(NT * sizeof(double));
    double *NB = (double *)malloc(NT * sizeof(double));
    double E[7] = {1., 10., 100., 1000., 10000., 100000., 1000000.}, D, dT;
    
    initialize_grids();
    initialize_H(-1.0);

    for (int e = 0; e < 1; e++) {

        D = 3.3e-8*pow(E[e], 1./3); // D = 1e28 (E/GeV)^(1/3) cm^2/s = 3.3e-8 (E/GeV)^(1/3) kpc^2/yr
        dT = dZ1 * dZ1 / D / 10.0; // dT <= (dR^2+dZ^2) / D ~ 3e4 yr
        // Initialize particle density and apply boundary conditions
        initialize_to_disk(ndis_C, dT);
        initialize_to_0(ndis_B);
        solve_diffusion_equation(ndis_C, ndis_B, NC, NB, D, dT);

        // output as txt file
        char fn1[100], fn2[100], fn3[100], fn4[100], name_body[100];
        snprintf(name_body, sizeof(name_body), "%dGeV_R%d_Z%d.txt", (int)E[e], (int)R[NR-1], (int)Z[NZ-1]);
        snprintf(fn1, sizeof(fn1), "ndis_C_%s", name_body);
        snprintf(fn2, sizeof(fn2), "ndis_B_%s", name_body);
        snprintf(fn3, sizeof(fn3), "NC_%s", name_body);
        snprintf(fn4, sizeof(fn4), "NB_%s", name_body);

        write_array2D_to_txt(fn1, ndis_C);
        write_array2D_to_txt(fn2, ndis_B);
        write_array1D_to_txt(fn3, NC);
        write_array1D_to_txt(fn4, NB);
    }

    // Free the dynamically allocated memory
    free(ndis_C);
    free(ndis_B);
    free(NC);
    free(NB);
    return 0;
}



