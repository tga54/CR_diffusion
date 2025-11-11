#include <stdlib.h>
#include <math.h>
#include "diffusion_func.h"
#include <string.h>  // for strncpy

const int NR1 = 200, NZ1 = 100;
const double dR1 = 0.05, dZ1 = 0.01, dR2 = 0.5, dZ2 = 0.1, R_disk = 10.0, Z_disk = 1.0, R_inj = 10.0;
const double sigma_pC = 255e-27, sigma_pB = 239e-27, sigma_pC2B = 48e-27;
const double c = 3e10, yr2s = 3.1536e7, ratio = 40.46285;
const double ratio_threshold = 0.04;
double R[NR], Z[NZ], dR[NR], dZ[NZ];

void initialize_grids_log(void){
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

void initialize_grids_linear(void){
    double Rc = (double)NR1 * dR1;
    double Zc = (double)NZ1 * dZ1;
    for (int i = 0; i < NR; i++) {
        if (i * dR1 <= Rc)
        {
            R[i] = i * dR1;
            dR[i] = dR1;
        } 
        else {
            R[i] = Rc + (i - NR1) * dR2; 
            dR[i] = dR2; 
        }
    }
    for (int k = 0; k < NZ; k++) {
        if (k * dZ1 <= Zc){
            Z[k] = k * dZ1;
            dZ[k] = dZ1;
        } 
        else{
            Z[k] = Zc + (k - NZ1) * dZ2;  
            dZ[k] = dZ2;
        }
    }
}

void initialize_to_disk(double *ndis, const double dT) {
    for (int i = 0; i < NR; i++)
        for (int j = 0; j < NZ; j++)
            ndis[i * NZ + j] = (R[i] <= R_inj && Z[j] <= Z_disk) ? 1.0 * dT : 0.0;
}

void initialize_to_src(double *ndis, const double dT) {
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NZ; j++){
	    if (Z[j] < Z_disk){
	        ndis[i * NZ + j] = 64.6 * pow(R[i], 2.35) * exp(- R[i] / 1.528 ) * dT;
	    }
        } 
    }
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

void initialize_H(double *ndis_H, const double nH) {  
// nH = 0.0: no decay; nH > 0: flat hydrogen distribution; nH = -1.0:we take the distribution of H_2 and HI from doi:10.1093/mnras/staa1017, and HII from doi: 10.1111/j.1365-2966.2004.08349.x 
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
                // ndis_H[i*NZ+j] = ndis_HII[i*NZ+j];
            }
        }
        free(ndis_H2); free(ndis_HI); free(ndis_HII);
    }
    else{
        perror("Error input of nH!");
        return;
    }
}

void initialize_D(double *D, const double Dd, const double Dh){
    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++){
            if (R[i] <= R_disk && Z[j] <= Z_disk){
                D[i * NZ + j] = Dd;
            }
            else{
                D[i * NZ + j] = Dh;
            }
        }
    }
}

void write_array_to_bin(const char *filename, const double *arr, const int size) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("File opening failed");
        return;
    }
    fwrite(arr, sizeof(double), size, file);
    fclose(file);
}

void solve_diffusion_equation(double *ndis_C, double *ndis_B, double *ndis_H, double *NC, double *NB, double *D, const double dT, double *ss_time) {
    double *ndis_C_temp = malloc(NR * NZ * sizeof(double));
    double *ndis_B_temp = malloc(NR * NZ * sizeof(double));
    for (int t = 0; t < NT; t++) {
        if (t * 10 % NT == 0){
            printf("Time step is: %d ! \n", t);
        }
        // printf("Step %d\n", t);  // Add this to see progress
        NC[t] = 0.0;
        NB[t] = 0.0;
        // Parallelize the outer loops over the spatial grid
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < NR - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                // double R = j * dR;
                double dD_dR = (D[j * NZ + k] - D[(j - 1) * NZ + k]) / dR[j] / 2.0 + (D[(j + 1) * NZ + k] - D[j * NZ + k]) / dR[j + 1] / 2.0;
                double dD_dZ = (D[j * NZ + k] - D[j * NZ + k - 1]) / dZ[k] / 2.0 + (D[j * NZ + k + 1] - D[j * NZ + k]) / dZ[k + 1] / 2.0;
                double dn_dR_C = (ndis_C[j * NZ + k] - ndis_C[(j - 1) * NZ + k]) / dR[j] / 2.0 + (ndis_C[(j + 1) * NZ + k] - ndis_C[j * NZ + k]) / dR[j+1] / 2.0;
                double d2n_dR2_C = (2.0 / (dR[j] * (dR[j] + dR[j+1]))) * ndis_C[(j - 1) * NZ + k] - (2.0 / (dR[j] * dR[j+1])) * ndis_C[j * NZ + k] + (2.0 / (dR[j+1] * (dR[j] + dR[j+1]))) * ndis_C[(j + 1) * NZ + k];
                double dn_dZ_C = (ndis_C[j * NZ + k] - ndis_C[j * NZ + k - 1]) / dZ[k] / 2.0 + (ndis_C[j * NZ + k + 1] - ndis_C[j * NZ + k]) / dZ[k+1] / 2.0;
                double d2n_dZ2_C = (2.0 / (dZ[k] * (dZ[k] + dZ[k+1]))) * ndis_C[j * NZ + k - 1] - (2.0 / (dZ[k] * dZ[k+1])) * ndis_C[j * NZ + k] + (2.0 / (dZ[k+1] * (dZ[k] + dZ[k+1]))) * ndis_C[j * NZ + k + 1];
                double decay_C = ndis_C[j * NZ + k] * ndis_H[j * NZ + k] * sigma_pC * c * dT * yr2s;
                ndis_C_temp[j * NZ + k] = ndis_C[j * NZ + k] + dT *((dD_dR * dn_dR_C + dD_dZ * dn_dZ_C) + D[j * NZ + k] * (d2n_dR2_C + dn_dR_C / R[j] + d2n_dZ2_C))- decay_C;  // Carbon number density distribution at this time step


                double d2n_dR2_B = (2.0 / (dR[j] * (dR[j] + dR[j+1]))) * ndis_B[(j - 1) * NZ + k] - (2.0 / (dR[j] * dR[j+1])) * ndis_B[j * NZ + k] + (2.0 / (dR[j+1] * (dR[j] + dR[j+1]))) * ndis_B[(j + 1) * NZ + k];
                double dn_dR_B = (ndis_B[j * NZ + k] - ndis_B[(j - 1) * NZ + k]) / dR[j] / 2.0 + (ndis_B[(j + 1) * NZ + k] - ndis_B[j * NZ + k]) / dR[j+1] / 2.0;
                double dn_dZ_B = (ndis_B[j * NZ + k] - ndis_B[j * NZ + k - 1]) / dZ[k] / 2.0 + (ndis_B[j * NZ + k + 1] - ndis_B[j * NZ + k]) / dZ[k+1] / 2.0;
                double d2n_dZ2_B = (2.0 / (dZ[k] * (dZ[k] + dZ[k+1]))) * ndis_B[j * NZ + k - 1] - (2.0 / (dZ[k] * dZ[k+1])) * ndis_B[j * NZ + k] + (2.0 / (dZ[k+1] * (dZ[k] + dZ[k+1]))) * ndis_B[j * NZ + k + 1];
                double decay_B = ndis_B[j * NZ + k] * ndis_H[j * NZ + k] * sigma_pB * c * dT * yr2s;
                ndis_B_temp[j * NZ + k] = ndis_B[j * NZ + k] + dT * ((dD_dR * dn_dR_B + dD_dZ * dn_dZ_B) + D[j * NZ + k] * (d2n_dR2_B + dn_dR_B / R[j] + d2n_dZ2_B)) - decay_B + decay_C / sigma_pC * sigma_pC2B; // Boron number density distribution at this time step
            }
        }

        apply_boundary_conditions(ndis_B_temp);
        apply_boundary_conditions(ndis_C_temp);
        // Parallelize the update step for ndis array
        #pragma omp parallel for collapse(2) schedule(static)
        // #pragma omp parallel for collapse(2) reduction(+:NC[t], NB[t])
        for (int j = 0; j < NR; j++) {
            for (int k = 0; k < NZ; k++) {
                if (R[j] <= R_inj && Z[k] <= Z_disk) {
	            // ndis_C[j * NZ + k] = 64.6 * pow(R[j], 2.35) * exp(- R[j] / 1.528 ) * dT + ndis_C_temp[j * NZ + k];
                    ndis_C[j * NZ + k] = 1.0 * dT + ndis_C_temp[j * NZ + k];  // ndis_C is the Carbon density at this time step after injection
                } 
                else {
                    ndis_C[j * NZ + k] = ndis_C_temp[j * NZ + k];
                }
                ndis_B[j * NZ + k] = ndis_B_temp[j * NZ + k];

                if (fabs(R[j] - 8.0) < 1e-6 && fabs(Z[k] - 0.0) < 1e-6){  // to avoid numericla residuals
                    NC[t] = ndis_C_temp[j * NZ];
                    NB[t] = ndis_B_temp[j * NZ];              
                }
            }
        }
    	double a = log10((double)t * dT);
        if (a - round(a) < 1e-10){
            char fn1[100], fn2[100];
                snprintf(fn1, sizeof(fn1), "ndis_C_1GV_T%d_injdisk_HI.bin", (int)a);
                snprintf(fn2, sizeof(fn2), "ndis_B_1GV_T%d_injdisk_HI.bin", (int)a);
            write_array_to_bin(fn1, ndis_C, NR*NZ);
            write_array_to_bin(fn2, ndis_B, NR*NZ);
        }
        if (t > (NT / 10)){            
            double ratio_C = (NC[t] - NC[t - 1]) * t / NC[t - 1];
            double ratio_B = (NB[t] - NB[t - 1]) * t / NC[t - 1];
            if (ratio_C < ratio_threshold && ratio_B < ratio_threshold){ 
                *ss_time = (double)t * dT;        
                // printf("Steady state at %e year\n", *ss_time);  
                // break;        // stop calculation when it gets to steady state
            }
            // if (t == NT - 1){
            //     // printf("The system did not get steady state !");
            //     *ss_time = t * dT;
	        // }
        }
    
    }
    free(ndis_C_temp);
    free(ndis_B_temp);
}

void thomas_solve(const double *a, const double *b, const double *c, const double *d, double *x, int n)
{
    if (n < 2) {
        if (n == 1) x[0] = d[0] / b[0];
        return;
    }

    // Allocate temporary arrays for modified coefficients
    double *cp = (double *)malloc(n * sizeof(double));
    double *dp = (double *)malloc(n * sizeof(double));

    // Modify the first coefficients
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    // Forward sweep
    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * cp[i - 1];
        if (denom == 0.0) {
            fprintf(stderr, "Zero pivot in Thomas algorithm at i=%d.\n", i);
            free(cp); free(dp);
            return;
        }
        cp[i] = (i < n - 1) ? c[i] / denom : 0.0;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }

    // Back substitution
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }

    free(cp);
    free(dp);
}

void CN_scheme(double *ndis, double *src, double *ndis_H, double *dT_tau, double *temp, double sigma, double D, double dT) {
    // define ADI coefficients
    double alpha_R[NR], alpha_Z[NZ];
    for (int i = 0; i < NR - 1; i++) alpha_R[i] = D * dT / (dR[i] + dR[i + 1]) ;
    for (int i = 0; i < NZ - 1; i++) alpha_Z[i] = D * dT / (dZ[i] + dZ[i + 1]) ;

    #pragma omp parallel for collapse(2) 
    for (int j = 0; j < NR; j++) {            
        for (int k = 0; k < NZ; k++) {
            dT_tau[j * NZ + k] = dT * yr2s * ndis_H[j * NZ + k] * sigma * c;
        }
    }

    //  Step 1: implicit in R, explicit in Z 
    #pragma omp parallel for default(none) shared(temp, ndis, dT_tau, alpha_Z, alpha_R, R, dR, dZ)
    for (int k = 1; k < NZ - 1; k++) {     
        double CN_Ra[NR], CN_Rb[NR], CN_Rc[NR], CN_Rd[NR], CN_Rx[NR];
        for (int j = 1; j < NR - 1; j++) {
            CN_Rd[j] = alpha_Z[k] / dZ[k] * ndis[j * NZ + k - 1] + (1.0 - alpha_Z[k] * (dZ[k + 1] + dZ[k]) / dZ[k + 1] / dZ[k]) * ndis[j * NZ + k] + alpha_Z[k] / dZ[k + 1] * ndis[j * NZ + k + 1] - ndis[j * NZ + k] * dT_tau[j * NZ + k] / 4.0;
        }                   // right hand side
        for (int j = 1; j < NR - 1; j++){     // cylinderical coordinate system  
            CN_Ra[j] = - alpha_R[j] * (1.0 / dR[j] - 0.5 / R[j]);
            CN_Rb[j] = 1.0 + alpha_R[j] * (dR[j + 1] + dR[j]) / dR[j + 1] / dR[j] + dT_tau[j * NZ + k] / 4.0;
            CN_Rc[j] = - alpha_R[j] * (1.0 / dR[j + 1] + 0.5 / R[j]);
        }
        CN_Ra[0] = 0.0;
        CN_Rb[0] = 1.0;
        CN_Rc[0] = -1.0;
        CN_Rd[0] = 0.0;
        thomas_solve(CN_Ra, CN_Rb, CN_Rc, CN_Rd, CN_Rx, NR - 1);
        for (int j = 0; j < NR; j++) temp[j * NZ + k] = CN_Rx[j];
    }
    apply_boundary_conditions(temp);

    //  Step 2: implicit in Z, explicit in R 
    #pragma omp parallel for default(none) shared(temp, ndis, dT_tau, alpha_Z, alpha_R, R, dR, dZ, src)
    for (int j = 1; j < NR - 1; j++){
        double CN_Za[NZ], CN_Zb[NZ], CN_Zc[NZ], CN_Zd[NZ], CN_Zx[NZ];
        for (int k = 1; k < NZ - 1; k++){
            CN_Zd[k] = alpha_R[j] * (1.0 / dR[j] - 0.5 / R[j]) * temp[(j-1) * NZ + k] + (1.0 - alpha_R[j] * (dR[j + 1] + dR[j]) / dR[j + 1] / dR[j]) * temp[j * NZ + k] + alpha_R[j] * (1.0 / dR[j + 1] + 0.5 / R[j]) * temp[(j+1) * NZ + k] - temp[j * NZ + k] * dT_tau[j * NZ + k] / 4.0;
        }
        for (int k = 1; k < NZ - 1; k++){
            CN_Za[k] = - alpha_Z[k] / dZ[k];
            CN_Zb[k] = 1.0 + alpha_Z[k] * (dZ[k + 1] + dZ[k]) / dZ[k + 1] / dZ[k] + dT_tau[j * NZ + k] / 4.0;
            CN_Zc[k] = - alpha_Z[k] / dZ[k + 1];
        }

        CN_Za[0] = 0.0;
        CN_Zb[0] = 1.0;
        CN_Zc[0] = -1.0;
        CN_Zd[0] = 0.0;
        thomas_solve(CN_Za, CN_Zb, CN_Zc, CN_Zd, CN_Zx, NZ - 1);
        for (int k = 0; k < NZ; k++) ndis[j * NZ + k] = CN_Zx[k] + src[j * NZ + k];
    }
    apply_boundary_conditions(ndis);
}

void solve_diffusion_equation_CN(double *ndis_C, double *ndis_B, double *ndis_H, double *NC, double *NB, double D, const double dT, double *ss_time){
    double *src_C = malloc(NR * NZ * sizeof(double));
    double *src_B = malloc(NR * NZ * sizeof(double));
    double *temp = malloc(NR * NZ * sizeof(double));
    double *dT_tau = malloc(NR * NZ * sizeof(double));

    initialize_to_disk(src_C, dT);
    initialize_to_0(temp);
    initialize_to_0(dT_tau);
    for (int t = 0; t < NT; t++){
        CN_scheme(ndis_C, src_C, ndis_H, dT_tau, temp, sigma_pC, D, dT);
        for (int j = 0; j < NR; j++){
            for (int k = 0; k < NZ; k++){
                src_B[j * NZ + k] = ndis_C[j * NZ + k] * ndis_H[j * NZ + k] * sigma_pC2B * c * dT * yr2s;
            }
        }
        CN_scheme(ndis_B, src_B, ndis_H, dT_tau, temp, sigma_pB, D, dT);
        double a = log10((double)t * dT);
        if (a - round(a) < 1e-10){
            char fn1[100], fn2[100];
                snprintf(fn1, sizeof(fn1), "ndis_C_1GV_T%d_CN_Htot_dt100.bin", (int)a);
                snprintf(fn2, sizeof(fn2), "ndis_B_1GV_T%d_CN_Htot_dt100.bin", (int)a);
            write_array_to_bin(fn1, ndis_C, NR*NZ);
            write_array_to_bin(fn2, ndis_B, NR*NZ);
	    }
        NC[t] = ndis_C[160 * NZ];
        NB[t] = ndis_B[160 * NZ];
        if (t > (NT / 10)){            
            double ratio_C = (NC[t] - NC[t - 1]) * t / NC[t - 1];
            double ratio_B = (NB[t] - NB[t - 1]) * t / NC[t - 1];
            if (ratio_C < ratio_threshold && ratio_B < ratio_threshold){ 
                *ss_time = (double)t * dT;        
                // printf("Steady state at %e year\n", *ss_time);  
                // break;        // stop calculation when it gets to steady state
            }
            // if (t == NT - 1){
            //     // printf("The system did not get steady state !");
            //     *ss_time = t * dT;
	        // }
        }
    }
    free(temp);
    free(dT_tau);
    free(src_C);
    free(src_B);
}

void write_array2D_to_txt(const char *filename, const double *arr) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return;
    }

    for (int i = 0; i < NR; i++) {
        for (int j = 0; j < NZ; j++) {
            fprintf(file, "%f,", arr[i * NZ + j]);
        }
        fputc('\n', file);  // new row
    }
    fclose(file);
}

void write_array1D_to_txt(const char *filename, const double *arr) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening output file");
        return;
    }
    // double ind_end = log10((double)NT) * 10.0;  

    for (int i = 0; i < NT; i++) {
        // double ind = pow(10.0, i / 10.0 + 1.0); //write 10 values for every order of magnitude, and start from N[10];
        fprintf(file, "%f,",  arr[i]);
    }
    fclose(file);
}



void write_array_header_to_bin(const char *filename, const double *arr, const int size, const double dT, const double *ss_time, const double Dd, const double Dh, const double gas_type, const char *grid_type) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("File opening failed");
        return;
    }

    // Fill header information
    BinHeader header;
    header.NR_BH = NR;
    header.NZ_BH = NZ;
    header.dR_BH = dR1;      // or dR[0] if grid spacing varies
    header.dZ_BH = dZ1;      // likewise
    header.dT_BH = dT;      // set dynamically if available
    header.Dd_BH = Dd;
    header.Dh_BH = Dh;
    header.ss_time_BH = *ss_time;      // set dynamically if available
    header.ndis_H_BH = gas_type;
    header.R_disk_BH = R_disk;
    char description[128];
    snprintf(description, sizeof(description), "Cosmic ray diffusion simulation with %s grid.", grid_type);
    strncpy(header.description_BH, description, sizeof(header.description_BH));
    header.description_BH[sizeof(header.description_BH)-1] = '\0'; // ensure null termination

    // Write header first
    fwrite(&header, sizeof(BinHeader), 1, file);

    // Then write array data
    fwrite(arr, sizeof(double), size, file);

    fclose(file);
}
