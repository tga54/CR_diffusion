#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include the OpenMP library

// large arrays like ndis, temp and N need dynamically allocate memory
const int NR = 501, NZ = 501, NT = 1000000;
const double D = 1e-6, R_inner_disk = 8.0, Z_disk = 0.5, R_outer_disk = 30.0;
const double dR = 0.1, dZ = 0.1, dT = 1000.0; // D = 1e-6 kpc²/yr, dT = 1000 yr, dR = dZ = 0.1 kpc

const double sigma_pC = 60.0, sigma_pB = 11.0, c = 3e10, yr2s = 3.1536e7; // cross section unit: mb, speed unit: cm / s
const double R_max = NR * dR, Z_max = NZ * dZ;

// Function to initialize particle density
void initialize_C(double **ndis) {
    #pragma omp parallel for // Parallelize the initialization loop
    for (int i = 0; i < NR; i++) {
        double R_temp = i * dR;
        for (int j = 0; j < NZ; j++) {
            double Z_temp = j * dZ;
            if (R_temp <= R_inner_disk && Z_temp <= Z_disk) {
           // if (i == 0 && j == 0) {
                ndis[i][j] = 1.0;
            } else {
                ndis[i][j] = 0.0;
            }
        }
    }
}

void initialize_B(double **ndis) {
    #pragma omp parallel for // Parallelize the initialization loop
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            ndis[i][j] = 0.0;
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

// Distribution of atom H
void initialize_H(double **ndis_H){
    double **ndis_H2 = (double **)malloc(NR * sizeof(double *));
    double **ndis_HI = (double **)malloc(NR * sizeof(double *));
    double **ndis_HII = (double **)malloc(NR * sizeof(double *));
    for (int i = 0; i < NR; i++) {
        ndis_H2[i] = (double *)malloc(NZ * sizeof(double));
        ndis_HI[i] = (double *)malloc(NZ * sizeof(double));
        ndis_HII[i] = (double *)malloc(NZ * sizeof(double));
    }

    for (int i = 0; i < NR; i++) {
        double R_temp = i * dR;
        for (int j = 0; j < NZ; j++) {
            double Z_temp = j * dZ;
            if (R_temp <= R_inner_disk) {
                ndis_H2[i][j] = 0.6 * exp(- Z_temp / 0.1 ) ;  // cm**-3
                ndis_HI[i][j] = 0.16 * exp(- Z_temp / 0.43) ;
                ndis_HII[i][j] = 0.025 * exp(- Z_temp) * exp(-pow(R_temp/20,2)) + 0.2 * exp(-Z_temp /0.15) * exp(-pow((R_temp-4)/2,2)) + 1e-3 / pow((1 + pow(R_temp/6,2)), 0.645);
                ndis_H[i][j] = 10*(ndis_H2[i][j] + ndis_HI[i][j] + ndis_HII[i][j]);
            } else if (R_temp >= R_inner_disk && R_temp <= R_outer_disk) {
                ndis_H2[i][j] = 0.02 * exp(- Z_temp / 0.3 ) ;  // cm**-3
                ndis_HI[i][j] = 0.06 * exp(- Z_temp / (0.3 * exp((R_temp - 9.8) / 9.8)));
                ndis_HII[i][j] = 0.025 * exp(- Z_temp) * exp(-pow(R_temp/20,2)) + 0.2 * exp(-Z_temp /0.15) * exp(-pow((R_temp-4)/2,2)) + 1e-3 / pow((1 + pow(R_temp/6,2)), 0.645);
                ndis_H[i][j] = 10*(ndis_H2[i][j] + ndis_HI[i][j] + ndis_HII[i][j]);

            } else {
                ndis_H2[i][j] = 0.0;
                ndis_HI[i][j] = 0.0;
                ndis_HII[i][j] = 1e-3 / pow((1 + pow(R_temp/6,2)) ,0.645);
                ndis_H[i][j] = 10*(ndis_H2[i][j] + ndis_HI[i][j] + ndis_HII[i][j]);
            }
        }
    }

    for (int i = 0; i < NR; i++) {
    free(ndis_H2[i]);
    free(ndis_HI[i]);
    free(ndis_HII[i]);
    }
    free(ndis_H2);
    free(ndis_HI);
    free(ndis_HII);
} 

// Function to solve the diffusion equation
void solve_diffusion_equation(double **ndis_C, double **ndis_B, double **ndis_H, double **temp, double NC[NT], double NB[NT]) {
    for (int t = 0; t < NT; t++) {
        // printf("Step %d\n", t);  // Add this to see progress
        // NC[t] = 0.0;
        // NB[t] = 0.0;
        // Parallelize the outer loops over the spatial grid
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int j = 1; j < NR - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                double R = j * dR;
                double dn_dR_C = (ndis_C[j + 1][k] - ndis_C[j - 1][k]) / (2 * dR);
                double d2n_dR2_C = (ndis_C[j + 1][k] + ndis_C[j - 1][k] - 2 * ndis_C[j][k]) / (dR * dR);
                double d2n_dZ2_C = (ndis_C[j][k + 1] + ndis_C[j][k - 1] - 2 * ndis_C[j][k]) / (dZ * dZ);
                double decay_C = ndis_C[j][k] * ndis_H[j][k] * sigma_pC * c * 1e-27 * dT * yr2s;

                temp[j][k] = ndis_C[j][k] + dT * D * (d2n_dR2_C + dn_dR_C / R + d2n_dZ2_C) - decay_C;
                double dn_dR_B = (ndis_B[j + 1][k] - ndis_B[j - 1][k]) / (2 * dR);
                double d2n_dR2_B = (ndis_B[j + 1][k] + ndis_B[j - 1][k] - 2 * ndis_B[j][k]) / (dR * dR);
                double d2n_dZ2_B = (ndis_B[j][k + 1] + ndis_B[j][k - 1] - 2 * ndis_B[j][k]) / (dZ * dZ);
                double decay_B = ndis_B[j][k] * ndis_H[j][k] * sigma_pB * c * 1e-27 * dT * yr2s;
                ndis_B[j][k] = ndis_B[j][k] + dT * D * (d2n_dR2_B + dn_dR_B / R + d2n_dZ2_B) - decay_B + decay_C;
            }
        }

        apply_boundary_conditions(ndis_B);
        apply_boundary_conditions(temp);

        // Parallelize the update step for ndis array
        #pragma omp parallel for collapse(2) reduction(+:NC[t], NB[t])
        for (int j = 0; j < NR; j++) {
            double R = j * dR;  // Radial distance from center
            for (int k = 0; k < NZ; k++) {
                double Z = k * dZ;

                if (R <= R_inner_disk && Z <= Z_disk) {
                    ndis_C[j][k] = 1.0 + temp[j][k];  // Constant injection in disk region
                } 
                else {
                    ndis_C[j][k] = temp[j][k];
                }
                if (j < NR - 1){
                    // NC[t] += (temp[j][k] + temp[j + 1][k]) / 2.0 * fabs(R);
                    // NB[t] += (ndis_B[j][k] + ndis_B[j + 1][k]) / 2.0 * fabs(R);                
                }
            }
        }
        NC[t] = temp[80][0];      
        NB[t] = ndis_B[80][0];  
    }
}

int main(void) {
    // Set number of threads to 10
    omp_set_num_threads(10);

    // Dynamically allocate memory for ndis and temp
    double **ndis_C = (double **)malloc(NR * sizeof(double *));
    double **ndis_B = (double **)malloc(NR * sizeof(double *));
    double **ndis_H = (double **)malloc(NR * sizeof(double *));
    double **temp = (double **)malloc(NR * sizeof(double *));
    for (int i = 0; i < NR; i++) {
        ndis_C[i] = (double *)malloc(NZ * sizeof(double));
        ndis_B[i] = (double *)malloc(NZ * sizeof(double));
        ndis_H[i] = (double *)malloc(NZ * sizeof(double));
        temp[i] = (double *)malloc(NZ * sizeof(double));
    }
    double *NC = (double *)malloc(NT * sizeof(double));
    double *NB = (double *)malloc(NT * sizeof(double));

 
    // Initialize particle density and apply boundary conditions
    initialize_C(ndis_C);
    initialize_B(ndis_B);
    initialize_H(ndis_H);
    apply_boundary_conditions(ndis_C);
    apply_boundary_conditions(ndis_B);

    // Solve the diffusion equation
   solve_diffusion_equation(ndis_C, ndis_B, ndis_H, temp, NC, NB);

    // file writing

    char filename1[50],filename2[50],filename3[50],filename4[50],filename5[50];
    snprintf(filename1, sizeof(filename1), "NC_t%d_R%d_Z%d_D1_H1.txt", NT, (int)R_max, (int)Z_max);
    snprintf(filename2, sizeof(filename2), "NB_t%d_R%d_Z%d_D1_H1.txt", NT, (int)R_max, (int)Z_max);
    snprintf(filename3, sizeof(filename3), "ndisC_t%d_R%d_Z%d_D1_H1.txt", NT, (int)R_max, (int)Z_max);
    snprintf(filename4, sizeof(filename4), "ndisB_t%d_R%d_Z%d_D1_H1.txt", NT, (int)R_max, (int)Z_max);
    snprintf(filename5, sizeof(filename5), "ndisH_t%d_R%d_Z%d_D1_H1.txt", NT, (int)R_max, (int)Z_max);


    // write total number of particles to file1
    FILE *file1 = fopen(filename1, "w");
    if (file1 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    FILE *file2 = fopen(filename2, "w");
    if (file2 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    for (int i = 0; i < NT; i++) {
        fprintf(file1, "%f,", NC[i]);
        fprintf(file2, "%f,", NB[i]);
    }
    fclose(file1);  // Close file after writing
    fclose(file2);  // Close file after writing


    // Write the final density distribution to a file
    FILE *file3 = fopen(filename3, "w");
    if (file3 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    FILE *file4 = fopen(filename4, "w");
    if (file4 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    FILE *file5 = fopen(filename5, "w");
    if (file5 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            fprintf(file3, "%f,", ndis_C[i][j]);
            fprintf(file4, "%f,", ndis_B[i][j]);
            fprintf(file5, "%f,", ndis_H[i][j]);
        }
        fprintf(file3, "\n");
        fprintf(file4, "\n");
        fprintf(file5, "\n");
    }
    fclose(file3);  // Close the file after writing
    fclose(file4);  // Close the file after writing
    fclose(file5);  // Close the file after writing

    // Free the dynamically allocated memory
    for (int i = 0; i < NR; i++) {
        free(ndis_C[i]);
        free(ndis_B[i]);
        free(ndis_H[i]);
        free(temp[i]);
    }
    free(ndis_C);
    free(ndis_B);
    free(ndis_H);
    free(temp);
    free(NC);
    free(NB);
    return 0;
}

