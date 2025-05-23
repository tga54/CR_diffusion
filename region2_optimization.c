#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Include the OpenMP library

// large arrays like ndis, temp and N need dynamically allocate memory
const int NR = 1381, NZ = 499, NT = 10000;
const double dR1 = 0.01, dZ1 = 0.01, dR2 = 0.5, dZ2 = 0.5; 
const double R_inj = 16.0, Z_inj = 1.0, Rc = 10.0, Zc = 1.0, ratio_threshold = 1e-6; 

const double sigma_pC = 60.0, sigma_pB = 11.0, c = 3e10, yr2s = 3.1536e7, ratio = 40.46285; // cross section unit: mb, speed unit: cm / s, ratio is M_sun / M_p / (pc / cm)**3
const double dRc = Rc / dR1, dZc = Zc / dZ1;

// Function to initialize Carbon number density distribution: ndis = dN / dV = dN / (R dR dz d\theta), due to rotational symmetry, we ignore d\theta and focus on R-z plane
void initialize_C(double *ndis_C, double R[NR], double Z[NZ], double dT) {
    // #pragma omp parallel for // Parallelize the initialization loop
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            if (R[i] <= R_inj && Z[j] <= Z_inj) {
                ndis_C[i * NZ + j] = 1.0 * dT;
            } else {
                ndis_C[i * NZ + j] = 0.0;
            }
        }
    }
}

void initialize_B(double *ndis_B) {
    // #pragma omp parallel for // Parallelize the initialization loop
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            ndis_B[i * NZ + j] = 0.0;
        }
    }
}

// Function to apply boundary conditions
void apply_boundary_conditions(double *ndis) {
    // #pragma omp parallel for
    for (int i = 0; i < NZ; i++) {
        ndis[i] = ndis[NZ + i];   // reflection boundary at ndis[:,0]
        ndis[(NR - 1) * NZ + i] = 0.0; // absorption boundary at ndis[:,NR - 1]
    }
    // #pragma omp parallel for
    for (int j = 0; j < NR; j++) {
        ndis[j * NZ ] = ndis[j * NZ + 1]; // reflection boundary at ndis[0,:]
        ndis[(j + 1) * NZ - 1] = 0.0;// absorption boundary at ndis[NR - 1,:]
    }
}

// Distribution of Hydrogen gas
void initialize_H(double *ndis_H, double R[NR], double Z[NZ]){
    double *ndis_H2 = malloc(NR * NZ * sizeof(double));
    double *ndis_HI = malloc(NR * NZ * sizeof(double));
    double *ndis_HII = malloc(NR * NZ * sizeof(double));

    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            double r0 = pow( R[i]*R[i] + Z[j]*Z[j], 0.5);
            ndis_H2[i * NZ + j] = ratio * 2200. / 4. / 45. * exp(-12./R[i] - R[i] / 1.5) / pow(cosh(Z[j] / 2. / 0.045), 2);
            ndis_HI[i * NZ + j] = ratio * 53. / 4. / 85. * exp(-4./R[i] - R[i] / 7) / pow(cosh(Z[j] / 2. / 0.085), 2);
            ndis_HII[i * NZ + j] = 0.00015 * (1. + 3.7*log(1 + r0 / 20.) / (r0/20.) - 1.0277);
            ndis_H[i * NZ + j] = ndis_H2[i * NZ + j] + ndis_HI[i * NZ + j] + ndis_HII[i * NZ + j];
        }
    }
    free(ndis_H2);
    free(ndis_HI);
    free(ndis_HII);
} 


// Function to solve the diffusion equation
void solve_diffusion_equation(double *ndis_C, double *ndis_B, double *ndis_H, double *ndis_C_temp, double *ndis_B_temp, double *NC, double *NB, double D, double dT, double R[NR], double Z[NZ], double dR[NR], double dZ[NZ]) {

    for (int t = 0; t < NT; t++) {
        // printf("Step %d\n", t);  // Add this to see progress
        NC[t] = 0.0;
        NB[t] = 0.0;
        // Parallelize the outer loops over the spatial grid
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < NR - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                // double R = j * dR;
                double d2n_dR2_C = (2.0 / (dR[j] * (dR[j] + dR[j+1]))) * ndis_C[(j - 1) * NZ + k] - (2.0 / (dR[j] * dR[j+1])) * ndis_C[j * NZ + k] + (2.0 / (dR[j+1] * (dR[j] + dR[j+1]))) * ndis_C[(j + 1) * NZ + k];
                double dn_dR_C = (ndis_C[j * NZ + k] - ndis_C[(j - 1) * NZ + k]) / dR[j] / 2.0 + (ndis_C[(j + 1) * NZ + k] - ndis_C[j * NZ + k]) / dR[j+1] / 2.0;
                double d2n_dZ2_C = (2.0 / (dZ[k] * (dZ[k] + dZ[k+1]))) * ndis_C[j * NZ + k - 1] - (2.0 / (dZ[k] * dZ[k+1])) * ndis_C[j * NZ + k] + (2.0 / (dZ[k+1] * (dZ[k] + dZ[k+1]))) * ndis_C[j * NZ + k + 1];
                double decay_C = ndis_C[j * NZ + k] * ndis_H[j * NZ + k] * sigma_pC * c * 1e-27 * dT * yr2s;
                ndis_C_temp[j * NZ + k] = ndis_C[j * NZ + k] + dT * D * (d2n_dR2_C + dn_dR_C / R[j] + d2n_dZ2_C) - decay_C;  // Carbon number density distribution at this time step

                double d2n_dR2_B = (2.0 / (dR[j] * (dR[j] + dR[j+1]))) * ndis_B[(j - 1) * NZ + k] - (2.0 / (dR[j] * dR[j+1])) * ndis_B[j * NZ + k] + (2.0 / (dR[j+1] * (dR[j] + dR[j+1]))) * ndis_B[(j + 1) * NZ + k];
                double dn_dR_B = (ndis_B[j * NZ + k] - ndis_B[(j - 1) * NZ + k]) / dR[j] / 2.0 + (ndis_B[(j + 1) * NZ + k] - ndis_B[j * NZ + k]) / dR[j+1] / 2.0;

                double d2n_dZ2_B = (2.0 / (dZ[k] * (dZ[k] + dZ[k+1]))) * ndis_B[j * NZ + k - 1] - (2.0 / (dZ[k] * dZ[k+1])) * ndis_B[j * NZ + k] + (2.0 / (dZ[k+1] * (dZ[k] + dZ[k+1]))) * ndis_B[j * NZ + k + 1];
                double decay_B = ndis_B[j * NZ + k] * ndis_H[j * NZ + k] * sigma_pB * c * 1e-27 * dT * yr2s;
                ndis_B_temp[j * NZ + k] = ndis_B[j * NZ + k] + dT * D * (d2n_dR2_B + dn_dR_B / R[j] + d2n_dZ2_B) - decay_B + decay_C; // Boron number density distribution at this time step
            }
        }

        apply_boundary_conditions(ndis_B_temp);
        apply_boundary_conditions(ndis_C_temp);

        // Parallelize the update step for ndis array
        #pragma omp parallel for collapse(2) schedule(static)
        // #pragma omp parallel for collapse(2) reduction(+:NC[t], NB[t])
        for (int j = 0; j < NR; j++) {
            for (int k = 0; k < NZ; k++) {
                if (R[j] <= R_inj && Z[k] <= Z_inj) {
                    ndis_C[j * NZ + k] = 1.0 * dT + ndis_C_temp[j * NZ + k];  // ndis_C is the Carbon density at this time step after injection
                    ndis_B[j * NZ + k] = ndis_B_temp[j * NZ + k];
                } 
                else {
                    ndis_C[j * NZ + k] = ndis_C_temp[j * NZ + k];
                    ndis_B[j * NZ + k] = ndis_B_temp[j * NZ + k];
                }
                if (j == 800 && k == 0){
                    NC[t] = ndis_C_temp[j * NZ];
                    NB[t] = ndis_B_temp[j * NZ];              
                }
            }
        }
        if (t > 0){            
            double ratio_C = (NC[t] - NC[t - 1]) / dT;
            double ratio_B = (NB[t] - NB[t - 1]) / dT;
            if (ratio_C < ratio_threshold && ratio_B < ratio_threshold){ 
                double ss_time = (double)t * dT;        
                printf("Steady state at %e year\n", ss_time);  
                break;        // stop calculation when it gets to steady state
            }
        }
    }
}

int main(void) {

    // Set number of threads to 10
    omp_set_num_threads(10);

    // Dynamically allocate memory for ndis and temp
    double *ndis_C = malloc(NR * NZ * sizeof(double));
    double *ndis_B = malloc(NR * NZ * sizeof(double));
    double *ndis_C_temp = malloc(NR * NZ * sizeof(double));
    double *ndis_B_temp = malloc(NR * NZ * sizeof(double));
    double *ndis_H = malloc(NR * NZ * sizeof(double));
    double *NC = (double *)malloc(NT * sizeof(double));
    double *NB = (double *)malloc(NT * sizeof(double));

    double R[NR], Z[NR], dR[NR], dZ[NZ];

    for (int i = 0; i < NR; i++) {
        if (i * dR1 <= Rc)
        {
            R[i] = i * dR1;
            dR[i] = dR1;
        } 
        else {
            R[i] = Rc + (i - (int)dRc) * dR2;
            dR[i] = dR2;
        }
    }
    for (int k = 0; k < NZ; k++) {
        if (k * dZ1 <= Zc){
            Z[k] = k * dZ1;
            dZ[k] = dZ1;
        } 
        else{
            Z[k] = Zc + (k - (int)dZc) * dZ2;
            dZ[k] = dZ2;
        }
    }

// Grid generation




    // Solve the diffusion equation
    char filename1[50],filename2[50],filename3[50],filename4[50];

    double E[7] = {1, 10, 100, 1000, 10000, 100000, 1000000}, D, dT;
    
    // double E[7] = {1, 10, 100, 1000, 10000 }, D, dT;
    initialize_H(ndis_H, R, Z);

    for (int e = 0; e < 1; e++) {

        D = 3.3e-8*pow(E[e], 1./3); // D = 1e28 (E/GeV)^(1/3) cm^2/s = 3.3e-8 (E/GeV)^(1/3) kpc^2/yr
        dT = dR1 * dR1 / D / 10.0; // dT <= (dR^2+dZ^2) / D ~ 3e4 yr
        // Initialize particle density and apply boundary conditions
        initialize_C(ndis_C, R, Z, dT);
        initialize_B(ndis_B);
        apply_boundary_conditions(ndis_C);
        apply_boundary_conditions(ndis_B);
        solve_diffusion_equation(ndis_C, ndis_B, ndis_H, ndis_C_temp, ndis_B_temp, NC, NB, D, dT, R, Z, dR, dZ);

        snprintf(filename1, sizeof(filename1), "NC_R%d_Z%d_D%dGeV_1.txt", (int)R[NR-1], (int)Z[NZ-1], (int)E[e]);
        snprintf(filename2, sizeof(filename2), "NB_R%d_Z%d_D%dGeV_1.txt", (int)R[NR-1], (int)Z[NZ-1], (int)E[e]);
        snprintf(filename3, sizeof(filename3), "ndisC_R%d_Z%d_D%dGeV_1.txt", (int)R[NR-1], (int)Z[NZ-1], (int)E[e]);
        snprintf(filename4, sizeof(filename4), "ndisB_R%d_Z%d_D%dGeV_1.txt", (int)R[NR-1], (int)Z[NZ-1], (int)E[e]);


        // Write the final density distribution to a file
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
        for (int i = 0; i < NT; i++) {
            fprintf(file1, "%f,", NC[i]);
            fprintf(file2, "%f,", NB[i]);

        }
        for (int i = 0; i < NR; i++) {
            for (int j = 0; j < NZ; j++) {
                fprintf(file3, "%f,", ndis_C[i * NZ + j]);
                fprintf(file4, "%f,", ndis_B[i * NZ + j]);
            }
            fprintf(file3, "\n");
            fprintf(file4, "\n");

        }
        fclose(file1);  // Close the file after writing
        fclose(file2);  // Close the file after writing
        fclose(file3);  // Close the file after writing
        fclose(file4);  // Close the file after writing
    }



    // Free the dynamically allocated memory
    free(ndis_C);
    free(ndis_B);
    free(ndis_C_temp);
    free(ndis_B_temp);
    free(ndis_H);
    free(NC);
    free(NB);
    return 0;
}




