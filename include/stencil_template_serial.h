/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

// ============================================================
// function prototypes

int initialize (int,
                char **,
                unsigned int *,
                int *,
                int *,
                int *,
                int **,
                double *,
                double **,
                int *,
                int *);

int memory_release (double *, int *);


extern int inject_energy (  int,
                            int,
                            int *,
                            double,
                            unsigned int [2],
                            double *,
                            double *); // Timestamp

extern int update_plane (   int,
                            unsigned int [2],
                            double *,
		                    double *,
                            double *); //Timestamp

extern int get_total_energy( unsigned int [2],
                             double *,
                             double *);


int dump (  const double *, 
            unsigned int [2], 
            const char *, 
            double *, 
            double * );

int memory_allocate ( unsigned int [2],
		                  double **);

int initialize_sources( unsigned int [2],
			                  int ,
			                  int **);

void export_and_plot(double *, double *, double *,int);

// ============================================================
// function definition for inline functions

inline int inject_energy (  int periodic,
                            int Nsources,
			                int *Sources,
			                double energy,
			                unsigned int mysize[2],
                            double *plane,
                            double *time_spent)
{

    #define IDX( i, j ) ( (j)*(mysize[_x_]+2) + (i) )
    
    // Start timing
    double begin = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < Nsources; s++) {
        
        unsigned x = Sources[2*s];
        unsigned y = Sources[2*s+1];
        plane[IDX(x, y)] += energy;

        if ( periodic )
            {
                if ( x == 1 )
                    plane[IDX(mysize[_x_]+1, y)] += energy;
                if ( x == mysize[_x_] )
                    plane[IDX(0, y)] += energy;
                if ( y == 1 )
                    plane[IDX(x, mysize[_y_]+1)] += energy;
                if ( y == mysize[_y_] )
                    plane[IDX(x, 0)] += energy;
            }
    }
    // End timing
    double end = omp_get_wtime();
    *time_spent = end - begin;
    //printf("Time spent injecting energy: %f seconds\n", *time_spent);

    #undef IDX
    return 0;
}

inline int update_plane (   int periodic,
                            unsigned int size[2],
			                double *old,
                            double *new,
                            double *time_spent)

/*
 * calculate the new energy values
 * the old plane contains the current data, the new plane
 * will store the updated data
 *
 * NOTE: in parallel, every MPI task will perform the
 *       calculation for its patch
 *
 */
{
    register const unsigned fxsize = size[_x_]+2;
    //register const unsigned fysize = size[_y_]+2;
    register const unsigned xsize = size[_x_];
    register const unsigned ysize = size[_y_];

    #define IDX( i, j ) ( (j)*fxsize + (i) )

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    //
    // HINT: in any case, this loop is a good candidate
    //       for openmp parallelization

    // Start timing
    double begin = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int j = 1; j <= ysize; j++)
        for (unsigned int i = 1; i <= xsize; i++)
        {
            // five-points stencil formula
            // simpler stencil with no explicit diffusivity
            // always conserve the smoohed quantity
            // alpha here mimics how much "easily" the heat
            // travels
            
            double alpha = 0.6;
            double result = old[ IDX(i,j) ] *alpha;
            double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]) / 4.0 * (1-alpha);
            double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]) / 4.0 * (1-alpha);
            result += (sum_i + sum_j );
            
            // implentation from the derivation of
            // 3-points 2nd order derivatives
            // however, that should depends on an adaptive
            // time-stepping so that given a diffusivity
            // coefficient the amount of energy diffused is
            // "small"
            // however the implicit methods are not stable

            /*
            #define alpha_guess 0.5     // mimic the heat diffusivity

            double alpha = alpha_guess;
            double sum = old[IDX(i,j)];
            double result = old[ IDX(i,j) ] *alpha;

            int   done = 0;
            do
                {                
                    double sum_i = alpha * (old[IDX(i-1, j)] + old[IDX(i+1, j)] - 2*sum);
                    double sum_j = alpha * (old[IDX(i, j-1)] + old[IDX(i, j+1)] - 2*sum);
                    result = sum + ( sum_i + sum_j);
                    double ratio = fabs((result-sum)/(sum!=0? sum : 1.0));
                    done = ( (ratio < 2.0) && (result >= 0) );    // not too fast diffusion and
                                                                    // not so fast that the (i,j)
                                                                    // goes below zero energy
                    alpha /= 2;
                }
            while ( !done );
            */

            new[ IDX(i,j) ] = result;
        }
    
    // End timing
    double end = omp_get_wtime();
    *time_spent = end - begin;
    //printf("Time spent updating: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
        
    if ( periodic )
    /*
    * propagate boundaries if they are periodic
    *
    * NOTE: when is that needed in distributed memory, if any?
    */
    {
        // Vertical boundaries top <-> bottom
        for (unsigned int i = 1; i <= xsize; i++ )
            {
                new[ i ] = new[ IDX(i, ysize) ];
                new[ IDX(i, ysize+1) ] = new[ i ];
            }
        // Horizontal boundaries left <-> right
        for (unsigned int j = 1; j <= ysize; j++ )
            {
                new[ IDX( 0, j) ] = new[ IDX(xsize, j) ];
                new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
            }
    }

    #undef IDX
    return 0;
}

inline int get_total_energy(unsigned int size[2],
                            double *plane,
                            double *energy)
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{
    register const int xsize = size[_x_];
    
    #define IDX( i, j ) ( (j)*(xsize+2) + (i) )

    #if defined(LONG_ACCURACY)    
        long double totenergy = 0;
    #else
        double totenergy = 0;    
    #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4

    #pragma omp parallel for collapse(2) reduction(+:totenergy) schedule(static)
    for (unsigned int j = 1; j <= size[_y_]; j++ )
        for (unsigned int i = 1; i <= size[_x_]; i++ )
            totenergy += plane[ IDX(i, j) ];
    
    #undef IDX

    *energy = (double)totenergy;
    return 0;
}
                            
