#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <omp.h> // Include the OpenMP library
// large arrays like ndis, temp and N need dynamically allocate memory
const int NX = 101, NT = 100000001;
const double dx = 1, ratio_threshold = 1e-6; 

const double X_max = NX * dx;

// Function to initialize particle density
void initialize(double *ndis) {
    // #pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        if (i == 0) {
            ndis[i] = 1.0;
        } else {
            ndis[i] = 0.0;
        }
    }
}

void initialize_to_0(double *ndis) {
    // #pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        ndis[i] = 0.0;
    }
}

// Function to apply boundary conditions
void apply_boundary_conditions(double *ndis) {
    // #pragma omp parallel for
    ndis[NX - 1] = 0.0;
    ndis[0] = ndis[1]; // !!! Important, since X=0 is not only the injection source but also a reflection boundary, do not forget this boundary condition.!!!
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

// Function to solve the diffusion equation
void solve_equation(double *ndis, double *temp, double D, double v, double dT) {
    for (int t = 0; t < NT; t++) {
        // printf("Step %d\n", t);  // Add this to see progress
        // #pragma omp parallel for
        for (int i = 1; i < NX - 1; i++) {
            double d2n_dx2 = (ndis[i + 1] + ndis[i - 1] - 2 * ndis[i]) / (dx * dx);
            double dn_dx = (ndis[i + 1] - ndis[i - 1]) / (2 * dx);
            temp[i] = ndis[i] + dT * (D * d2n_dx2 - v * dn_dx);
        }
        apply_boundary_conditions(temp);
        // #pragma omp parallel for
        for (int i = 0; i < NX; i++){
            ndis[i] = temp[i];
        }
        // ndis[0] += 1.0 * dT;

        int count = 0; 
        int T = t + 0; // range of int/_int32 is only ~ (-2e9, 2e9), so use long long int whose range is ~ (-9e18, 9e18)
        while (T % 10 == 0 && T != 0){ // check if T is power of 10, note that T can not be 0, other wise it will be infinite loop
            T = T / 10;
            count +=1;
        }
            if (T == 1){
                char filename1[50];
                snprintf(filename1, sizeof(filename1), "ndis_Pe1_t_gf.bin");
                write_array_to_bin(filename1, ndis, NX);
            }
    }
}



int main(void){

    // Dynamically allocate memory for ndis and temp
    double *ndis = (double *)malloc(NX * sizeof(double));
    double *temp = (double *)malloc(NX * sizeof(double));

    for (int e = 0; e < 1; e++) {
        double D = 10.0; // D = 1e28 (E/GeV)^(1/3) cm^2/s = 3.3e-8 (E/GeV)^(1/3) kpc^2/yr
        double v = 1.0;
        double dT = fmin(dx * dx / D / 10.0, dx / v / 10.0); // CFL condition: dT <= (dR^2+dZ^2) / D and dT <= dR / v 
        // double v = 0.0;
        // double dT = dx * dx / D / 10.0; // CFL condition: dT <= (dR^2+dZ^2) / D and dT <= dR / v 
        // Initialize particle density and apply boundary conditions
        initialize(ndis);
        initialize_to_0(temp);

        // Solve the diffusion equation
        solve_equation(ndis, temp, D, v, dT);

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

