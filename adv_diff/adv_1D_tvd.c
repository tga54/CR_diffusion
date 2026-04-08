#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include the OpenMP library
// large arrays like ndis, temp and N need dynamically allocate memory
const int NX = 20004, NT = 1000001;
const double dx = 1, ratio_threshold = 1e-6; 

const double X_max = NX * dx;

// Function to initialize particle density
void initialize(double *ndis) {
    #pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        if (i == NX / 2) ndis[i] = 1.0;
        else ndis[i] = 0.0;
    }
}

void initialize_to_0(double *ndis) {
    #pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        ndis[i] = 0.0;
    }
}

// Function to apply boundary conditions
void apply_boundary_conditions(double *ndis) {
    ndis[NX - 1] = 0.0;
    ndis[NX - 2] = 0.0; // for TVD scheme, we need to set ndis[NX-2] = 0.0, otherwise it will cause out of bound access when calculating r_ip at i = NX-2
    ndis[0] = 0.0; 
    ndis[1] = 0.0; // for TVD scheme, we need to set ndis[1] = 0.0, otherwise it will cause out of bound access when calculating r_ip at i = 1
}

void write_array_to_bin(const char *filename, const double *arr, const int size) {
    FILE *file = fopen(filename, "ab");
    if (!file) {
        perror("File opening failed");
        return;
    }
    fwrite(arr, sizeof(double), size, file);
    fclose(file);
}

void solve_equation_TVD(double *ndis, double *temp, double D, double v, double dT) {
    #pragma omp parallel
    {
        for (int t = 0; t < NT; t++) {
            #pragma omp for nowait
            for (int i = 2; i < NX - 2; i++) {
                double d2n_dx2 = (ndis[i + 1] + ndis[i - 1] - 2 * ndis[i]) / (dx * dx);
                double flux_ip = 0.0; // i+1/2
                double flux_im = 0.0; // i-1/2

                // ----- i+1/2 -----
                double r_ip = (ndis[i] - ndis[i-1]) / (ndis[i+1] - ndis[i] + 1e-12);
                double phi_ip = (r_ip + fabs(r_ip)) / (1.0 + fabs(r_ip)); // van Leer

                double slope_ip = phi_ip * (ndis[i+1] - ndis[i]);
                double n_ip = ndis[i] + 0.5 * slope_ip;

                flux_ip = v * n_ip;

                // ----- i-1/2 -----
                double r_im = (ndis[i-1] - ndis[i-2]) / (ndis[i] - ndis[i-1] + 1e-12);
                double phi_im = (r_im + fabs(r_im)) / (1.0 + fabs(r_im));

                double slope_im = phi_im * (ndis[i] - ndis[i-1]);
                double n_im = ndis[i-1] + 0.5 * slope_im;

                flux_im = v * n_im;
                
                double adv_term = -(flux_ip - flux_im) / dx;

                temp[i] = ndis[i] + dT * (D * d2n_dx2 + adv_term);
            }
            #pragma omp barrier

            #pragma omp single
            {
                apply_boundary_conditions(temp);
                double *swap = ndis;
                ndis = temp;
                temp = swap;
                ndis[NX / 2] += 1.0; // source term at the center, note that this source term is added after the boundary condition, so it will not be affected by the boundary condition.

                int count = 0; 
                int T = t + 0; // range of int/_int32 is only ~ (-2e9, 2e9), so use long long int whose range is ~ (-9e18, 9e18)
                while (T % 10 == 0 && T != 0){ // check if T is power of 10, note that T can not be 0, other wise it will be infinite loop
                    T = T / 10;
                    count +=1;
                }
                if (T == 1){
                    char filename1[50];
                    snprintf(filename1, sizeof(filename1), "ndis_Pe001_ci_TVD.bin");
                    write_array_to_bin(filename1, ndis, NX);
                }
            }
        }
    }
}


int main(void){
    // Set number of threads to the number of CPUs
    int ncore = omp_get_num_procs(); 
    omp_set_num_threads(10);
    // Dynamically allocate memory for ndis and temp
    double *ndis = (double *)malloc(NX * sizeof(double));
    double *temp = (double *)malloc(NX * sizeof(double));

    for (int e = 0; e < 1; e++) {
        double D = 1000.0; // D = 1e28 (E/GeV)^(1/3) cm^2/s = 3.3e-8 (E/GeV)^(1/3) kpc^2/yr
        double v = 1.0;
        // double dT = fmin(dx * dx / D / 10.0, dx / v / 10.0); // CFL condition: dT <= (dR^2+dZ^2) / D and dT <= dR / v 
        double dT = 0.0001;
        // double v = 0.0;
        // double dT = dx * dx / D / 10.0; // CFL condition: dT <= (dR^2+dZ^2) / D and dT <= dR / v 
        // Initialize particle density and apply boundary conditions
        initialize(ndis);
        initialize_to_0(temp);

        // Solve the diffusion equation
        solve_equation_TVD(ndis, temp, D, v, dT);

        // file writing

        // char filename1[50], filename2[50];
     
        // // snprintf(filename1, sizeof(filename1), "ndis_Pe0.1.bin");
        // // write_array_to_bin(filename1, ndis, NX);
    }

    // Free the dynamically allocated memory

    free(ndis);
    free(temp);
    return 0;
}

