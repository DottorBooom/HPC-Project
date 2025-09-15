/**
 * @file stencil_serial.c
 * @brief Serial implementation of 2D heat diffusion using stencil computation
 * 
 * This program simulates heat diffusion on a 2D grid using a 5-point stencil pattern.
 * The simulation uses double buffering to efficiently update grid values and supports
 * both periodic and non-periodic boundary conditions. Energy is injected at random
 * heat sources throughout the simulation.
 * 
 * Key features:
 * - OpenMP parallelization for performance
 * - Energy conservation verification
 * - Load balancing analysis
 * - Comprehensive timing measurements
 * - Support for periodic boundary conditions
 * 
 * Usage: ./stencil_serial [options]
 * Options:
 *   -x size      Grid width (default: 1000)
 *   -y size      Grid height (default: 1000)
 *   -n iter      Number of iterations (default: 99)
 *   -e sources   Number of heat sources (default: 1)
 *   -E energy    Energy per source (default: 1.0)
 *   -f freq      Injection frequency (default: 0.0)
 *   -p flag      Periodic boundaries (default: 0)
 *   -o flag      Output energy at each step (default: 0)
 * 
 * @author Davide
 * @date 2024
 */

// filepath: /home/davide/Documents/Assignment/src/stencil_serial.c
#include "stencil_serial.h"

/**
 * @brief Main function - Heat diffusion simulation driver
 * 
 * Controls the entire heat diffusion simulation process including:
 * - Parameter initialization from command line arguments
 * - Memory allocation for simulation data structures
 * - Main simulation loop with energy injection and grid updates
 * - Performance measurement and load balancing analysis
 * - Energy conservation verification
 * - Results output and cleanup
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return 0 on successful completion, 1 on error
 */
int main(int argc, char **argv)
{

  // Simulation parameters
  int  Niterations;             // Total number of simulation iterations
  int  periodic;                // Flag for periodic boundary conditions
  unsigned int S[2];            // Grid dimensions [width, height]

  // Heat source configuration
  int     Nsources;             // Number of heat sources on the grid
  int    *Sources;              // Array of heat source coordinates (x,y pairs)
  double  energy_per_source;    // Energy amount injected per source

  // Simulation planes for double buffering
  double *planes[2];            // Two planes: current (OLD) and updated (NEW)
  
  // Energy tracking variables
  double injected_heat = 0;     // Total energy injected into the system

  // Timing and output control
  int injection_frequency;      // How often to inject energy (every N iterations)
  int output_energy_at_steps = 0; // Flag to print energy budget at each step
   
  int ret;                      // Return value for error checking

  double init_time, total_time; // Timing measurements

  // Start timing the initialization phase
  init_time = omp_get_wtime();
  
  /* Parse command line arguments and initialize simulation parameters */
  ret = initialize ( argc, argv, &S[0], &periodic, &Niterations,
	       &Nsources, &Sources, &energy_per_source, &planes[0],
	       &output_energy_at_steps, &injection_frequency );

  if (ret == 1)
  {
    printf("Error during memory allocation\n");
    return 1;
  }
  
  // Set initial plane index for double buffering
  int current = OLD;

  // Arrays for performance measurement and load balancing analysis
  double time_spent_updating [Niterations];  // Per-iteration computation times

  int nthreads = omp_get_num_threads();       // Number of OpenMP threads
  double thread_time [nthreads];              // Per-thread timing for load balancing

  // Initial energy injection (if frequency > 1, inject at start)
  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current]);

  // Complete initialization timing
  init_time = omp_get_wtime() - init_time;
  
  // Start timing the total simulation
  total_time = omp_get_wtime();

  // Main simulation loop
  for (int iter = 0; iter < Niterations; iter++)
  {

    /* Energy injection phase - inject energy into heat sources */
    if ( iter % injection_frequency == 0 )
    {
      inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current]);
      injected_heat += Nsources*energy_per_source;
    }
                  
    /* Grid update phase - apply 5-point stencil heat diffusion */
    update_plane(periodic, S, planes[current], planes[!current], &time_spent_updating[iter], thread_time);

    /* Energy budget output (if enabled) - for debugging and verification */
    if ( output_energy_at_steps )
    {
      double system_heat;
      get_total_energy( S, planes[!current], &system_heat);
              
      printf("step %d :: injected energy is %g, updated system energy is %g\n", 
              iter, injected_heat, system_heat );

      // Optional: dump current state to binary file for visualization
      //char filename[100];
      //sprintf( filename, "plane_%05d.bin", iter );
      //dump( planes[!current], S, filename, NULL, NULL );        
    }

    /* Plane swapping for double buffering - avoid data copying */
      current = !current;

  }
  total_time = omp_get_wtime() - total_time;

  /* Final energy conservation check - verify total energy in the system */
  double system_heat;
  get_total_energy( S, planes[current], &system_heat);
  printf("injected energy is %g, system energy is %g\n", injected_heat, system_heat );
  
  /* Clean up allocated memory */
  memory_release( planes[OLD], Sources );

  /* Performance analysis - calculate total computation time */
  double sumU = 0;

  #pragma omp parallel for reduction(+: sumU) schedule(static)
  for(int i = 0; i < Niterations; i++)
  {
    sumU += time_spent_updating[i];
  }

  /* Load balancing analysis for the last iteration */
  double max_time = thread_time[0], min_time = thread_time[0], avg_time = 0.0;
  for (int i = 0; i < nthreads; i++) {
      if (thread_time[i] > max_time) max_time = thread_time[i];
      if (thread_time[i] < min_time) min_time = thread_time[i];
      avg_time += thread_time[i];
  }
  avg_time /= nthreads;
  double load_imbalance = (max_time - min_time) / avg_time;
  
  /* Output comprehensive performance statistics */
  printf("Total time: %f\n", total_time);
  printf("Initialization time: %f\n", init_time);
  printf("Computation time: %f\n", sumU);
  printf("Computation/Total ratio: %.2f%%\n", (sumU/total_time)*100.0);
  printf("Other time (overhead): %f (%.2f%%)\n", 
    total_time - sumU, 
    ((total_time - sumU)/total_time)*100.0);
  printf("Load imbalance: %.2f\n", load_imbalance);

  /* Optional: export timing data for detailed analysis */
  //export_and_plot(time_spent_injecting, time_spent_updating, time_spent_simulation, Niterations);

  return 0;
}