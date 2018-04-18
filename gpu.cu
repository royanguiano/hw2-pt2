#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <vector>
using namespace std;

#if !defined density
#define density 0.0005
#endif

#if !defined mass
#define mass 0.01
#endif

#if !defined cutoff
#define cutoff 0.01
#endif

#if !defined min_r
#define min_r (cutoff/100)
#endif

#if !defined dt
#define dt      0.0005
#endif

#define NUM_THREADS 256

extern double size;
//
//  benchmarking program
//

// calculate particle's bin number
__device__ int calculateBinNum(particle_t &p, int binsPerSide)
{
    return ( floor(p.x/cutoff) + binsPerSide*floor(p.y/cutoff) );
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}
__global__ void setupParticleBin(particle_t * particles, int n, particle_t * bins, int numbins, int binsPerSide, int* binSizes) {
    // clear bins at each time step
    for (int m = 0; m < numbins; m++) {
        binSizes[m] = 0;
    }
    
    // place particles in bins
    for (int i = 0; i < n; i++) {
        int binNumber = calculateBinNum(particles[i],binsPerSide);
        int indexInBin = binSizes[binNumber];
        bins[binNumber*n + indexInBin] = particles[i];
        binSizes[binNumber] ++;
    }
}

__global__ void compute_forces_gpu(particle_t * particles, int n, particle_t * bins, int numbins, int binsPerSide, int* binSizes)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
    
    // find current particle's bin, handle boundaries
    int cbin = calculateBinNum( particles[tid], binsPerSide );
    int lowi = -1, highi = 1, lowj = -1, highj = 1;
    if (cbin < binsPerSide) lowj = 0;
    if (cbin % binsPerSide == 0) lowi = 0;
    if (cbin % binsPerSide == (binsPerSide-1)) highi = 0;
    if (cbin >= binsPerSide*(binsPerSide-1)) highj = 0;
    
    // apply nearby forces
    for (int i = lowi; i <= highi; i++) {
        for (int j = lowj; j <= highj; j++)
        {
            int nbin = cbin + i + binsPerSide*j;
            
            for (int indexInBin = 0; indexInBin < binSizes[nbin]; indexInBin++) {
                apply_force_gpu(particles[tid], bins[nbin*n + indexInBin]);
            }
            
            //for (int k = 0; k < bins[nbin].size(); k++ ) {
                //apply_force( local[p], *bins[nbin][k], &dmin, &davg, &navg);
                
            //    apply_force_gpu(particles[tid], particles[j]);
            //}
        }
    }

    //for(int j = 0 ; j < n ; j++) {
    //    apply_force_gpu(particles[tid], particles[j]);
    //}

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
	printf( "-s <filename> to specify the summary output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen(sumname,"a") : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    // create spatial bins (of size cutoff by cutoff)
    double size = sqrt( density*n );
    int binsPerSide = ceil(size/cutoff);
    int numbins = binsPerSide*binsPerSide;
    
    particle_t* bins = (particle_t *) malloc(n * sizeof(particle_t) * numbins);
    
    int* binSizes = (int *) malloc(numbins * sizeof(int));

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    
    particle_t * d_bins;
    cudaMalloc((void **) &d_bins, n * sizeof(particle_t) * numbins);
    
    int * d_binSizes;
    cudaMalloc((void **) &d_binSizes, sizeof(int) * numbins);

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        
        setupParticleBin <<< 1, 1 >>> (d_particles, n, d_bins, numbins, binsPerSide, d_binSizes);

	int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, d_bins, numbins, binsPerSide, d_binSizes);
        
        //
        //  move particles
        //
	move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    if (fsum)
	fprintf(fsum,"%d %lf \n",n,simulation_time);

    if (fsum)
	fclose( fsum );    
    free( particles );
    free( bins );
    cudaFree(d_particles);
    
    cudaFree(d_bins);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
