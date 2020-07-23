#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define MAX_ITER 100
#define THRESHOLD 1e-6

// Global Variables used across different functions
int number_of_points_global;
int number_of_threads_global;
int number_of_iterations_global;
int K_global;
int *data_points_global;
float *iter_centroids_global;
int *data_point_cluster_global;
int **iter_cluster_count_global;

// Defined global delta
double delta_global = THRESHOLD + 1;

void kmeans_openmp_thread(int *tid)
{
    int *id = (int *)tid;

    // Assigning data points range to each thread
    int data_length_per_thread = number_of_points_global / number_of_threads_global;
    int start = (*id) * data_length_per_thread;
    int end = start + data_length_per_thread;
    if (end + data_length_per_thread > number_of_points_global)
    {
        //To assign last undistributed points to this thread for computation, change end index to number_of_points_global
        end = number_of_points_global;
        data_length_per_thread = number_of_points_global - start;
    }

    printf("Thread ID:%d, start:%d, end:%d\n", *id, start, end);

    int i = 0, j = 0;
    double min_dist, current_dist;

    // Cluster id associated with each point
    int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));

    // Cluster location or centroid (x,y,z) coordinates for K clusters in a iteration
    float *cluster_points_sum = (float *)malloc(K_global * 3 * sizeof(float));

    // No. of points in a cluster for a iteration
    int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));

    // Start of loop
    int iter_counter = 0;
    while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER))
    {
        // Initialize cluster_points_sum or centroid to 0.0
        for (i = 0; i < K_global * 3; i++)
            cluster_points_sum[i] = 0.0;

        // Initialize number of points for each cluster to 0
        for (i = 0; i < K_global; i++)
            points_inside_cluster_count[i] = 0;

        for (i = start; i < end; i++)
        {
            //Assign these points to their nearest cluster
            min_dist = DBL_MAX;
            for (j = 0; j < K_global; j++)
            {
                current_dist = pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3] - (float)data_points_global[i * 3]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_points_global[i * 3 + 1]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_points_global[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i - start] = j;
                }
            }

            //Update local count of number of points inside cluster
            points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;

            // Update local sum of cluster data points
            cluster_points_sum[point_to_cluster_id[i - start] * 3] += (float)data_points_global[i * 3];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 1] += (float)data_points_global[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 2] += (float)data_points_global[i * 3 + 2];
        }

/*
    Update iter_centroids_global and iter_cluster_count_global after each thread arrival
    Supporting formula is
    (prev_iter_centroid_global * prev_iter_cluster_count + new_thread_cluster_points_sum) / (new_thread_cluster_count + prev_iter_cluster_count) 
*/
#pragma omp critical
        {
            for (i = 0; i < K_global; i++)
            {
                if (points_inside_cluster_count[i] == 0)
                {
                    printf("Unlikely situation!\n");
                    continue;
                }
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 1]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 2]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                
                iter_cluster_count_global[iter_counter][i] += points_inside_cluster_count[i];
            }
        }

/*
    Wait for all threads to arrive and execute for first thread only
    Delta is the sum of squared distance between centroid of previous and current iteration.
    Supporting formula is:
        delta = (iter1_centroid1_x - iter2_centroid1_x)^2 + (iter1_centroid1_y - iter2_centroid1_y)^2 + (iter1_centroid1_z - iter2_centroid1_z)^2 + (iter1_centroid2_x - iter2_centroid2_x)^2 + (iter1_centroid2_y - iter2_centroid2_y)^2 + (iter1_centroid2_z - iter2_centroid2_z)^2
    Update delta_global with new delta
*/
#pragma omp barrier
        if (*id == 0)
        {
            double temp_delta = 0.0;
            for (i = 0; i < K_global; i++)
            {
                temp_delta += (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]);
            }
            delta_global = temp_delta;
            number_of_iterations_global++;
        }

// Wait for all thread to arrive and update the iter_counter by +1
#pragma omp barrier
        iter_counter++;
    }
//End of loop

// Assign points to final choice for cluster centroids
    for (i = start; i < end; i++)
    {
        // Assign points to clusters
        data_point_cluster_global[i * 4] = data_points_global[i * 3];
        data_point_cluster_global[i * 4 + 1] = data_points_global[i * 3 + 1];
        data_point_cluster_global[i * 4 + 2] = data_points_global[i * 3 + 2];
        data_point_cluster_global[i * 4 + 3] = point_to_cluster_id[i - start];
        assert(point_to_cluster_id[i - start] >= 0 && point_to_cluster_id[i - start] < K_global);
    }
}

void kmeans_omp(int num_threads,
                    int N,
                    int K,
                    int *data_points,
                    int **data_points_cluster_id,
                    float **iter_centroids,
                    int *number_of_iterations)
{


    // Initialize global variables
    number_of_points_global = N;
    number_of_threads_global = num_threads;
    number_of_iterations_global = 0;
    K_global = K;
    data_points_global = data_points;

    *data_points_cluster_id = (int *)malloc(N * 4 * sizeof(int));   //Allocating space of 4 units each for N data points
    data_point_cluster_global = *data_points_cluster_id;

    /*
        Allocating space of 3K units for each iteration
        Since three dimensional data point and K number of clusters 
    */
    iter_centroids_global = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assigning first K points to be initial centroids
    int i = 0;
    for (i = 0; i < K; i++)
    {
        iter_centroids_global[i * 3] = data_points[i * 3];
        iter_centroids_global[i * 3 + 1] = data_points[i * 3 + 1];
        iter_centroids_global[i * 3 + 2] = data_points[i * 3 + 2];
    }

    // Print initial centroids
    for (i = 0; i < K; i++)
    {
        printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_centroids_global[i * 3], iter_centroids_global[i * 3 + 1], iter_centroids_global[i * 3 + 2]);
    }

    /*
        Allocating space for iter_cluster_count_global
        iter_cluster_count_global keeps the count of number of points in K clusters after each iteration
     */
    iter_cluster_count_global = (int **)malloc(MAX_ITER * sizeof(int *));
    for (i = 0; i < MAX_ITER; i++)
    {
        iter_cluster_count_global[i] = (int *)calloc(K, sizeof(int));
    }

    // Creating threads
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("Thread: %d created!\n", ID);
        kmeans_openmp_thread(&ID);
    }

    // Record number_of_iterations
    *number_of_iterations = number_of_iterations_global;

    // Record number of iterations and store iter_centroids_global data into iter_centroids
    int iter_centroids_size = (*number_of_iterations + 1) * K * 3;
    printf("Number of iterations :%d\n", *number_of_iterations);
    *iter_centroids = (float *)calloc(iter_centroids_size, sizeof(float));
    for (i = 0; i < iter_centroids_size; i++)
    {
        (*iter_centroids)[i] = iter_centroids_global[i];
    }

    // Print final centroids after last iteration
    for (i = 0; i < K; i++)
    {
        printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*number_of_iterations) * K + i) * 3], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 1], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 2]);
    }

}

void dataset_in(const char *dataset_filename, int *N, int **data_points)
{
	FILE *fin = fopen(dataset_filename, "r");
	fscanf(fin, "%d", N);
	*data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
    int i = 0;
	for (i = 0; i < (*N) * 3; i++)
	{
		fscanf(fin, "%d", (*data_points + i));
	}
	fclose(fin);
}

void clusters_out(const char *cluster_filename, int N, int *cluster_points)
{
	FILE *fout = fopen(cluster_filename, "w");
    int i = 0;
	for (i = 0; i < N; i++)
	{
		fprintf(fout, "%d %d %d %d\n",
				*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
				*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
	}
	fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int number_of_iterations, float *iter_centroids)
{
	FILE *fout = fopen(centroid_filename, "w");
    int i = 0;
	for (i = 0; i < number_of_iterations + 1; i++)
	{
        int j = 0;
		for (j = 0; j < K; j++)
		{
			fprintf(fout, "%f %f %f, ",
					*(iter_centroids + (i * K + j) * 3),		 //x coordinate
					*(iter_centroids + (i * K + j) * 3 + 1),  //y coordinate
					*(iter_centroids + (i * K + j) * 3 + 2)); //z coordinate
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

void main()
{

	//---------------------------------------------------------------------
	int N;					//no. of data points (input)
	int K;					//no. of clusters to be formed (input)
	int num_threads;		//no. of threads to be used (input)
	int* data_points;		//data points (input)
	int* cluster_points;	//clustered data points (to be computed)
	float* iter_centroids;			//centroids of each iteration (to be computed)
	int number_of_iterations;    //no of iterations performed by algo (to be computed)
	//---------------------------------------------------------------------

    char *dataset_filename = "dataset-10000.txt";

    printf("Enter No. of Clusters: ");
    scanf("%d", &K);
    printf("Enter No. of threads to be used: ");
    scanf("%d",&num_threads);

    printf("\nFollowing files should be in the same directory as of program\n");
    printf("1 for 10000 datapoints\n");
    printf("2 for 50000 datapoints\n");
    printf("3 for 100000 datapoints\n");
    printf("4 for 200000 datapoints\n");
    printf("5 for 400000 datapoints\n");
    printf("6 for 500000 datapoints\n");
    printf("7 for 600000 datapoints\n");
    printf("8 for 800000 datapoints\n");
    printf("9 for 1000000 datapoints\n");
    printf("\nEnter the number of dataset file to input: ");

    int x;
    scanf("%d",&x);

    switch (x)
    {
    case 1:
        dataset_filename = "dataset-10000.txt";
        break;
    case 2:
        dataset_filename = "dataset-50000.txt";
        break;
    case 3:
        dataset_filename = "dataset-100000.txt";
        break;
    case 4:
        dataset_filename = "dataset-200000.txt";
        break;
    case 5:
        dataset_filename = "dataset-400000.txt";
        break;
    case 6:
        dataset_filename = "dataset-500000.txt";
        break;
    case 7:
        dataset_filename = "dataset-600000.txt";
        break;
    case 8:
        dataset_filename = "dataset-800000.txt";
        break;
    case 9:
        dataset_filename = "dataset-1000000.txt";
        break;
    default:
        dataset_filename = "dataset-10000.txt";
        break;
    }


	double start_time, end_time;
	double computation_time;

	/*
        Function reads dataset_file and store data into data_points array. Each points have three consecutive indices associated into array.
        data_points array looks like : [pt_1_x, pt_1_y, pt_1_z, pt_2_x, pt_2_y, pt_2_z]
	*/

	dataset_in (dataset_filename, &N, &data_points);

    /*
        For one iteration and two clusters,
        iter_centroids array looks like : [iter_1_cluster_1_x, iter_1_cluster_1_y, iter_1_cluster_1_z, iter_1_cluster_2_x, iter_1_cluster_2_y, iter_1_cluster_2_z, iter_2_cluster_1_x, ...]
        Similarly the array extends further with more iterations
    */

	start_time = omp_get_wtime();
	kmeans_omp(num_threads, N, K, data_points, &cluster_points, &iter_centroids, &number_of_iterations);
	end_time = omp_get_wtime();

    // Creating filenames for different threads and different dataset
    char num_threads_char[3];
    snprintf(num_threads_char,10,"%d", num_threads);

    char file_index_char[2];
    snprintf(file_index_char,10,"%d", x);

    char cluster_filename[105] = "cluster_output_threads";
    strcat(cluster_filename,num_threads_char);
    strcat(cluster_filename,"_dataset");
    strcat(cluster_filename,file_index_char);
    strcat(cluster_filename,".txt");

    char centroid_filename[105] = "centroid_output_threads";
    strcat(centroid_filename,num_threads_char);
    strcat(centroid_filename,"_dataset");
    strcat(centroid_filename,file_index_char);
    strcat(centroid_filename,".txt");

	/*
        Clustered points are saved into cluster_filename.
        Each point is associated with the cluster index it belongs to.
        cluster_points array looks like : [pt_1_x, pt_1_y, pt_1_z, pt_1_cluster_index, pt_2_x, pt_2_y, pt_2_z, pt_2_cluster_index]
        Output file format:
            pt_1_x, pt_1_y, pt_1_z, pt_1_cluster_index
	*/
	clusters_out (cluster_filename, N, cluster_points);

    /*
        Centroid points are stored into centroid_filename.
        Each line in the file depicts the centroid coordinates of clusters after each iteration.
        Output file format:
            centroid_1_x, centroid_1_y, centroid_1_z, centroid_2_x, centroid_2_y, centroid_2_z
    */
	centroids_out (centroid_filename, K, number_of_iterations, iter_centroids);

    /*
        Computation time is stored in 'compute_time_openmp.txt'.
    */
   	computation_time = end_time - start_time;
	printf("Time Taken: %lf \n", computation_time);
    
	char time_file_omp[100] = "compute_time_openmp_threads";
    strcat(time_file_omp,num_threads_char);
    strcat(time_file_omp,"_dataset");
    strcat(time_file_omp,file_index_char);
    strcat(time_file_omp,".txt");

	FILE *fout = fopen(time_file_omp, "a");
	fprintf(fout, "%f\n", computation_time);
	fclose(fout);
    
	printf("Cluster Centroid point output file '%s' saved\n", centroid_filename);
    printf("Clustered points output file '%s' saved\n", cluster_filename);
    printf("Computation time output file '%s' saved\n", time_file_omp);
	
}