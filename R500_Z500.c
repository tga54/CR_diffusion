#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include the OpenMP library

// large arrays like ndis, temp and N need dynamically allocate memory
const int NR = 1001, NZ = 1001, NT = 1000000;
const double D = 500.0, R_max = 500000.0, Z_max = 500000.0, R_disk = 8000.0, Z_disk = 500.0;
const double dR = 500.0, dZ = 500.0, dT = 100.0; // D = 100 pc²/yr, dT = 100 yr, dR = dZ = 500 pc

// Function to initialize particle density
void initialize(double **ndis) {
    #pragma omp parallel for // Parallelize the initialization loop
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            if (i <= 16 && j <= 1) {
           // if (i == 0 && j == 0) {
                ndis[i][j] = 1.0;
            } else {
                ndis[i][j] = 0.0;
            }
        }
    }
}

// Function to apply boundary conditions
void apply_boundary_conditions(double **ndis) {
    #pragma omp parallel for
    for (int i = 0; i < NZ; i++) {
        ndis[NR - 1][i] = 0.0;
        ndis[0][i] = ndis[1][i];
    }
    #pragma omp parallel for
    for (int j = 0; j < NR; j++) {
        ndis[j][NZ - 1] = 0.0;
        ndis[j][0] = ndis[j][1];
    }
}

// Function to solve the diffusion equation
void solve_diffusion_equation(double **ndis, double **temp, double N[NT]) {
    // printf("Starting diffusion equation solver\n");
    for (int t = 0; t < NT; t++) {
        // printf("Step %d\n", t);  // Add this to see progress
        N[t] = 0.0;
        // Parallelize the outer loops over the spatial grid
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int j = 1; j < NR - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                double R_j = j * dR;
                double dn_dR = (ndis[j + 1][k] - ndis[j - 1][k]) / (2 * dR);
                double d2n_dR2 = (ndis[j + 1][k] + ndis[j - 1][k] - 2 * ndis[j][k]) / (dR * dR);
                double d2n_dZ2 = (ndis[j][k + 1] + ndis[j][k - 1] - 2 * ndis[j][k]) / (dZ * dZ);
                temp[j][k] = ndis[j][k] + dT * D * (d2n_dR2 + dn_dR / R_j + d2n_dZ2);
            }
        }

        apply_boundary_conditions(temp);

        // Parallelize the update step for ndis array
        #pragma omp parallel for collapse(2) reduction(+:N[t])
        for (int j = 0; j < NR; j++) {
            double R_j = j * dR;  // Radial distance from center

            for (int k = 0; k < NZ; k++) {
                if (j <= 16 && k <= 1) {
                    ndis[j][k] = 1.0 + temp[j][k];  // Constant injection in disk region
                } 
                else {
                    ndis[j][k] = temp[j][k];
                }
                if (j < NR - 1){
                    N[t] += (temp[j][k] + temp[j + 1][k]) / 2.0 * fabs(R_j);                
                }
            }
        }
    }
}

int main(void) {
    // Set number of threads to 10
    omp_set_num_threads(10);

    // Dynamically allocate memory for ndis and temp
    double **ndis = (double **)malloc(NR * sizeof(double *));
    double **temp = (double **)malloc(NR * sizeof(double *));
    for (int i = 0; i < NR; i++) {
        ndis[i] = (double *)malloc(NZ * sizeof(double));
        temp[i] = (double *)malloc(NZ * sizeof(double));
    }
    double *N = (double *)malloc(NT * sizeof(double));

 
    // Initialize particle density and apply boundary conditions
    initialize(ndis);


    apply_boundary_conditions(ndis);


    // Solve the diffusion equation
    solve_diffusion_equation(ndis, temp, N);

    // write total number of particles to file1
    FILE *file1 = fopen("N_RB_t1000000_R500_Z500_diskinj.txt", "w");
    if (file1 == NULL) {
        perror("Failed to open file");
        return 1;
    }

    for (int i = 0; i < NT; i++) {
        fprintf(file1, "%f,", N[i]);

    }
    fclose(file1);  // Close file after writing


    // Write the final density distribution to a file
    FILE *file2 = fopen("ndis_RB_t1000000_R500_Z500_diskinj.txt", "w");
    if (file2 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            fprintf(file2, "%f,", ndis[i][j]);
        }
        fprintf(file2, "\n");
    }
    fclose(file2);  // Close the file after writing

    // Free the dynamically allocated memory
    for (int i = 0; i < NR; i++) {
        free(ndis[i]);
        free(temp[i]);
    }
    free(ndis);
    free(temp);
    free(N);

    return 0;
}

