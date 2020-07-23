# Parallel-K-means-Clustering
Parallel implementation of K-means Clustering algorithm using OpenMp in C++

## How to run
K-means Clustering algorithm Execution Procedure:

### For Parallel Code:
1) Make sure that k-means_parallel.c and dataset file are in same directory. Unknown directories may cause segmentation fault.
2) Run the program with command on the terminal:
	gcc k-means_parallel.c -fopenmp -lm
3) Three inputs has to be given no. of clusters, no. of  threads, and dataset_index
4) Three output files are created in same directory, For 1 thread and dataset index 3:
centroid_output_threads1_dataset3.txt
cluster_output_threads1_dataset3.txt
compute_time_openmp_threads1_dataset3

### For Serial Code:
1) Make sure that k-means_sequential.c and dataset file are in same directory. Unknown directories may cause segmentation fault.
2) Run the program with command on the terminal:
	gcc k-means_sequential.c -fopenmp -lm
3) Two inputs has to be given no. of clusters and dataset_index
4) Three output files are created in same directory as, For dataset index 3:
centroid_output_dataset3.txt
cluster_output_dataset3.txt
compute_time_openmp_dataset3.txt

## Introduction

K-means algorithm is an iterative approach to partition the given dataset into K different subsets or clusters, where each data point belongs to only one cluster. It assigns data points into a cluster such that the sum of squared distances between data points is minimum. The lesser the variation we have inside the cluster, the more the homogenous clusters we get.

## Serial Algorithm and Complexity Analysis

The general algorithm we follow,

1. Provide the number of clusters K
2. Initialize K random centroids of the clusters from the dataset
3. Keep iterating the following steps until we get no change to the
centroid data point.
  a. Find the minimum distance from each data point to the
centroids.
  b. Assign the data point to the closest centroid.
  c. Compute the centroid of each cluster by taking the mean of data
points in that cluster.

This approach is known as Expectation-Maximization. Where the E-step is assigning different data points to the closest cluster in each iteration. Whereas M-step is calculating the mean of data points in the cluster after each iteration.

## Scope of parallelism

The clustering algorithm requires massive computations, with distance between each data point and each centroid being calculated. Since calculation of the appropriate centroid for each data point is independent of the others, the algorithm provides a good scope for parallelism. However, there is a bottleneck. The threads need to communicate among themselves to keep the centroid values updated, as more than one thread might try to access the same centroid point. In that case, it is imperative to ensure that both threads do not try to modify the centroid at the same time, as it might result in corrupted values and false sharing.

## Read More

For more information regarding parallelization strategies and results, read the report [K-means Clustering Report](https://github.com/ronak197/Parallel-K-means-Clustering/blob/master/K-means%20report.pdf). 

