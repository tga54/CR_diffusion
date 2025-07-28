#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>  // gethostname


#define PI 3.14159265358979323846
#define NR 601
#define NZ 601
#define n_particles_tot 100000
#define n_steps 10000000
#define dt 100

const double sigma_C = 255e-27, c_pcyr = 0.307, ratio = 40.46285, grid_per_order = 100.0, r_disk = 10000.0;
double R[NR], Z[NZ];

void initialize_to_0(double *arr, int arr_size) {
    for (int i = 0; i < arr_size; i++)
            arr[i] = 0.0;
}

void initialize_grids(void){
    for (int i = 0; i < NR; i++) {
        R[i] = pow(10.0, (double)i / grid_per_order);
    }
    for (int i = 0; i < NZ; i++) {
        Z[i] = pow(10.0, (double)i / grid_per_order);
    }
}

void initialize_pos(double *x, double *y, double *z, double r_disk, int n_particles, unsigned int *seed){
    for (int i = 0; i < n_particles; i++){
        double rand_R_squared = r_disk * r_disk * (double)rand_r(seed) / (double)RAND_MAX; // random radius of rdr ~ dr^2 from 0 to r_disk^2, here seed is already a pointer
        double rand_theta = 2.0 * PI *(double)rand_r(seed) / (double)RAND_MAX; // random angle d\theta from 0 to 2*pi
        x[i] = sqrt(rand_R_squared) * cos(rand_theta);
        y[i] = sqrt(rand_R_squared) * sin(rand_theta);
        z[i] = 0.0;
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
                ndis_H2[i*NZ+j] = ratio * 2200. / 4. / 45. * exp(-12000./R[i] - R[i] / 1500.) / pow(cosh(Z[j] / 2. / 45.), 2);
                ndis_HI[i*NZ+j] = ratio * 53. / 4. / 85. * exp(-4000./R[i] - R[i] / 7000.) / pow(cosh(Z[j] / 2. / 85.), 2);
                ndis_HII[i*NZ+j] = 0.00015 * (1. + 3.7*log(1 + r0 / 20000.) / (r0/20000.) - 1.0277);
                ndis_H[i*NZ+j] = ndis_H2[i*NZ+j] + ndis_HI[i*NZ+j] + ndis_HII[i*NZ+j];
                // ndis_H[i*NZ+j] = ndis_H2[i*NZ+j];
            }
        }
        free(ndis_H2); free(ndis_HI); free(ndis_HII);
    }
    else{
        perror("Error input of nH!");
        return;
    }
}

double interp_2D(double *ndis_H, double r_temp, double z_temp){

    double r1, r2, z1, z2, n11, n12, n21, n22, area, log_r_temp, log_z_temp; 
    log_r_temp = log10(r_temp);
    log_z_temp = log10(z_temp);
    int ind_r = (int)(log_r_temp * grid_per_order);
    int ind_z = (int)(log_z_temp * grid_per_order);

    if (ind_r < 0 || ind_r >= NR - 1 || ind_z < 0 || ind_z >= NZ - 1) {
        return 0.0;
    }
    else{
        r1 = log10(R[ind_r]);
        r2 = log10(R[ind_r + 1]);
        z1 = log10(Z[ind_z]);
        z2 = log10(Z[ind_z + 1]);
        n11 = ndis_H[ind_r * NZ + ind_z];
        n12 = ndis_H[ind_r * NZ + ind_z + 1];
        n21 = ndis_H[(ind_r + 1) * NZ + ind_z];
        n22 = ndis_H[(ind_r + 1) * NZ + ind_z + 1];
        area  = (r2 - r1) * (z2 - z1);
        double f = 1.0 / area * ( n11 * (r2 - log_r_temp) * (z2 - log_z_temp) + n21 * (log_r_temp - r1) * (z2 - log_z_temp) + n12 * (r2 - log_r_temp) * (log_z_temp - z1) + n22 * (log_r_temp - r1) * (log_z_temp - z1));
        if (f < 0.0){ return 0.0; }
        else{ return f; }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_particles_local = n_particles_tot / size;
    int record_ratio = 1000;
    int record_size_local = n_particles_local * n_steps / record_ratio;
    long long int record_size_global = (long long)n_particles_tot * n_steps / record_ratio;
    
    // Dynamically allocate memory for ndis and temp
    double l_scatter = c_pcyr * dt / 3.0;  // 10 pc
    double *x = malloc(n_particles_local * sizeof(double));
    double *y = malloc(n_particles_local * sizeof(double));
    double *z = malloc(n_particles_local * sizeof(double));
    double *gram_temp = malloc(n_particles_local * sizeof(double));
    double *ndis_H = malloc(NR * NZ * sizeof(double));

    double *gram_record_local = malloc(record_size_local * sizeof(double));
    double *time_record_local = malloc(record_size_local * sizeof(double));
    double *gram_record_global = NULL;
    double *time_record_global = NULL;
    if (rank == 0) {
        gram_record_global = malloc(record_size_global * sizeof(double));
        time_record_global = malloc(record_size_global * sizeof(double));
    }

    int ind = 0;
    unsigned int seed = time(NULL) + rank * 1337;
    initialize_grids();
    initialize_pos(x, y, z, r_disk, n_particles_local, &seed);
    initialize_to_0(gram_temp, n_particles_local);
    initialize_to_0(gram_record_local, record_size_local );
    initialize_to_0(time_record_local, record_size_local);
    initialize_H(ndis_H, -1.0);

    for (int t = 0; t < n_steps; t++){
        for (int i = 0; i < n_particles_local; i++) {
            double cos_theta_v = 2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0;
            double sin_theta_v = sqrt(1.0 - cos_theta_v * cos_theta_v);
            double phi_v = 2.0 * PI * (double)rand_r(&seed) / RAND_MAX;
            x[i] += l_scatter * sin_theta_v * cos(phi_v);
            y[i] += l_scatter * sin_theta_v * sin(phi_v);
            z[i] += l_scatter * cos_theta_v;

            double r_temp = sqrt(x[i]*x[i] + y[i]*y[i]);
            double z_temp = fabs(z[i]);
            double nH = interp_2D(ndis_H, r_temp, z_temp);

            gram_temp[i] += l_scatter * nH; 

            double dist = sqrt(pow(fabs(r_temp - 8000.0), 2.0) + pow(z_temp, 2.0));
            if (dist < l_scatter) {
		if (ind < record_size_local){
               	    gram_record_local[ind] = gram_temp[i];
                    time_record_local[ind] = (double)t;
		    ind++;
		}
		else{
		    printf("Local record size is too small !");
		    break;
		}
            }
        }
	if (rank == 0 && t % (n_steps / 10) == 0) {
            char hostname[256];
            gethostname(hostname, 256);
            printf("Rank 0 on %s: Time step %d / %d\n", hostname, t, n_steps);
        }
    }
    // Gather data to rank 0
    MPI_Gather(gram_record_local, record_size_local, MPI_DOUBLE, gram_record_global, record_size_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(time_record_local, record_size_local, MPI_DOUBLE, time_record_global, record_size_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     
 
    if (rank == 0) {
        FILE *file1 = fopen("MC_gram_disk_R10kpcZ0_mfp10pc_t1e9yr.bin", "wb");
        fwrite(gram_record_global, sizeof(double), record_size_global, file1);
        fclose(file1);

        FILE *file2 = fopen("MC_time_disk_R10kpcZ0_mfp10pc_t1e9yr.bin", "wb");
        fwrite(time_record_global, sizeof(double), record_size_global, file2);
        fclose(file2);
    }

    free(x); 
    free(y); 
    free(z); 
    free(gram_temp); 
    free(ndis_H);
    free(gram_record_local); 
    free(time_record_local);
    if (rank == 0) { 
        free(gram_record_global); 
        free(time_record_global); 
    }

    MPI_Finalize();

    return 0;
}
