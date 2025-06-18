#include <stdlib.h>
#include <math.h>
#include "diffusion_func.h"


const int NR1 = 200, NZ1 = 100;
const double dR1 = 0.05, dZ1 = 0.01, R_inj = 4.0, Z_inj = 1.0;
const double sigma_pC = 255.0, sigma_pB = 239.0, sigma_pC2B = 48.0;
const double c = 3e10, yr2s = 3.1536e7, ratio = 40.46285;
const double ratio_threshold = 1e-6;
double R[NR], Z[NZ], dR[NR], dZ[NZ], *ndis_H;

void initialize_grids(void){
    double Rc = (double)NR1 * dR1;
    double Zc = (double)NZ1 * dZ1;
    double indR = log10(Rc);
    double indZ = log10(Zc);
    
    for (int i = 0; i < NR; i++) {
        if (i * dR1 <= Rc)
        {
            R[i] = i * dR1;
            dR[i] = dR1;
        } 
        else {
            R[i] = pow(10.0, ((double)i - (int)NR1) / 400.0 + indR ); // 400 steps for every order of magnitude, i.e. R[i] = 10**1.0000, R[i+1] = 10**1.0025, R[i+2] = 10**1.005, R3 = 10**1.0075....... 
            dR[i] = R[i] - R[i - 1]; 
        }
        // printf("R is %.5f and dR is %.5f : ", R[i], dR[i]);
    }
    for (int k = 0; k < NZ; k++) {
        if (k * dZ1 <= Zc){
            Z[k] = k * dZ1;
            dZ[k] = dZ1;
        } 
        else{
            Z[k] = pow(10.0, ((double)k - (int)NZ1) / 100.0 + indZ ); //  100 steps for every order of magnitude: i.e. R[i] = 10**1.00, R[i+1] = 10**1.01, R[i+2] = 10**1.02, R[i+3] = 10**1.03....... 
            dZ[k] = Z[k] - Z[k - 1];
        }
        // printf("Z is %.5f and dZ is %.5f : ", Z[k], dZ[k]);
    }
}

void initialize_to_disk(double *ndis, const double dT) {
    for (int i = 0; i < NR; i++)
        for (int j = 0; j < NZ; j++)
            ndis[i * NZ + j] = (R[i] <= R_inj && Z[j] <= Z_inj) ? 1.0 * dT : 0.0;
}

void initialize_to_0(double *ndis) {
    for (int i = 0; i < NR; i++)
        for (int j = 0; j < NZ; j++)
            ndis[i * NZ + j] = 0.0;
}

void initialize_from_file(const char *filename, double *ndis) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening input file");
        return;
    }

    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            if (fscanf(file, "%lf,", &ndis[i * NZ + j]) != 1) {
                fprintf(stderr, "Error reading element [%d][%d]\n", i, j);
                fclose(file);
                return;
            }
        }
        // Consume the newline character after each row
        int c = fgetc(file);
        if (c != '\n' && c != EOF) {
            ungetc(c, file);  // put back if not newline or EOF
        }
    }
    fclose(file);
}


void apply_boundary_conditions(double *ndis) {
    for (int i = 0; i < NZ; i++) {
        ndis[i] = ndis[NZ + i];
        ndis[(NR - 1) * NZ + i] = 0.0;
    }
    for (int j = 0; j < NR; j++) {
        ndis[j * NZ] = ndis[j * NZ + 1];
        ndis[(j + 1) * NZ - 1] = 0.0;
    }
}

void initialize_H(const double nH) {  
// nH = 0.0: no decay; nH > 0: flat hydrogen distribution; nH = -1.0:we take the distribution of H_2 and HI from doi:10.1093/mnras/staa1017, and HII from doi: 10.1111/j.1365-2966.2004.08349.x 
    ndis_H = malloc(NR * NZ * sizeof(double)); 
    if (nH >= 0.0){
        for (int i = 0; i < NR; i++) {
            for (int j = 0; j < NZ; j++) {
                ndis_H[i*NZ+j] = nH;
            }
        }
    }
    else if (nH == -1.0){
        double *ndis_H2 = malloc(NR * NZ * sizeof(double));
        double *ndis_HI = malloc(NR * NZ * sizeof(double));
        double *ndis_HII = malloc(NR * NZ * sizeof(double));
        for (int i = 0; i < NR; i++) {
            for (int j = 0; j < NZ; j++) {
                double r0 = sqrt(R[i]*R[i] + Z[j]*Z[j]);
                ndis_H2[i*NZ+j] = ratio * 2200. / 4. / 45. * exp(-12./R[i] - R[i] / 1.5) / pow(cosh(Z[j] / 2. / 0.045), 2);
                ndis_HI[i*NZ+j] = ratio * 53. / 4. / 85. * exp(-4./R[i] - R[i] / 7) / pow(cosh(Z[j] / 2. / 0.085), 2);
                ndis_HII[i*NZ+j] = 0.00015 * (1. + 3.7*log(1 + r0 / 20.) / (r0/20.) - 1.0277);
                ndis_H[i*NZ+j] = ndis_H2[i*NZ+j] + ndis_HI[i*NZ+j] + ndis_HII[i*NZ+j];
            }
        }
        free(ndis_H2); free(ndis_HI); free(ndis_HII);
    }
    else{
        perror("Error input of nH!");
        return;
    }
}

void solve_diffusion_equation(double *ndis_C, double *ndis_B, double *NC, double *NB, const double D, const double dT) {
    double *ndis_C_temp = malloc(NR * NZ * sizeof(double));
    double *ndis_B_temp = malloc(NR * NZ * sizeof(double));
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
                ndis_B_temp[j * NZ + k] = ndis_B[j * NZ + k] + dT * D * (d2n_dR2_B + dn_dR_B / R[j] + d2n_dZ2_B) - decay_B + decay_C / sigma_pC * sigma_pC2B; // Boron number density distribution at this time step
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
                if (fabs(R[j] - 8.0) < 1e-6 && fabs(Z[k] - 0.0) < 1e-6){  // to avoid numericla residuals
                    NC[t] = ndis_C_temp[j * NZ];
                    NB[t] = ndis_B_temp[j * NZ];              
                }
            }
        }
        // if (t > 1000000){            
        //     double ratio_C = (NC[t] - NC[t - 1]) / dT;
        //     double ratio_B = (NB[t] - NB[t - 1]) / dT;
        //     if (ratio_C < ratio_threshold && ratio_B < ratio_threshold){ 
        //         double ss_time = (double)t * dT;        
        //         printf("Steady state at %e year\n", ss_time);  
        //         break;        // stop calculation when it gets to steady state
        //     }
        // }
    }
    free(ndis_C_temp);
    free(ndis_B_temp);
    free(ndis_H);
}

void write_array2D_to_txt(const char *filename, const double *ndis) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return;
    }

    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            fprintf(file, "%f,", ndis[i * NZ + j]);
        }
        fputc('\n', file);  // new row
    }
    fclose(file);
}

void write_array1D_to_txt(const char *filename, const double *N) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return;
    }
    double ind_end = log10((double)NT) * 10.0;  

    for (int i = 10; i < (int)ind_end; i++) {
        double ind = pow(10.0, i / 10.0 + 1.0); //write 10 values for every order of magnitude, and start from N[10];
        fprintf(file, "%d,%f\n", (int)ind, N[(int)ind]);
    }
    fclose(file);
}