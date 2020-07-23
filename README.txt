K-means Clustering algorithm Execution Procedure:

For Parallel Code:
1) Make sure that k-means_parallel.c and dataset file are in same directory. Unknown directories may cause segmentation fault.
2) Run the program with command on the terminal:
	gcc k-means_parallel.c -fopenmp -lm
3) Three inputs has to be given no. of clusters, no. of  threads, and dataset_index
4) Three output files are created in same directory, For 1 thread and dataset index 3:
centroid_output_threads1_dataset3.txt
cluster_output_threads1_dataset3.txt
compute_time_openmp_threads1_dataset3

For Serial Code:
1) Make sure that k-means_sequential.c and dataset file are in same directory. Unknown directories may cause segmentation fault.
2) Run the program with command on the terminal:
	gcc k-means_sequential.c -fopenmp -lm
3) Two inputs has to be given no. of clusters and dataset_index
4) Three output files are created in same directory as, For dataset index 3:
centroid_output_dataset3.txt
cluster_output_dataset3.txt
compute_time_openmp_dataset3.txt
