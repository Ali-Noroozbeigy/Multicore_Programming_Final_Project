#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "string.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BLOCKS 256
#define NUM_THREADS 256
#define NUM_CLASSES 6

cudaError_t compute_histogram(int* histo, char** input, int n);
void print_array(int* a);

__device__ int cuda_strcmp(const char* s1, const char* s2) {
	while (*s1 && *s2 && (*s1 == *s2)) {
		s1++;
		s2++;
	}
	return (*s1 - *s2);
}

__global__ void compute_hist_kernel(int* histo, char** input, int n) {

	__shared__ int histo_private[NUM_CLASSES];

	if (threadIdx.x < NUM_CLASSES) histo_private[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	int stride = blockDim.x * gridDim.x;

	while (i < n) {
	
		char *s1 = "elephant";
		char *s2 = input[i];
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[0]), 1);
		
		s1 = "lion";
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[1]), 1);
		
		s1 = "zebra";
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[2]), 1);
		
		s1 = "monkey";
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[3]), 1);		

		s1 = "tiger";
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[4]), 1);
		
		
		s1 = "leopard";
		while (*s1 && *s2 && (*s1 == *s2)) {
			s1++;
			s2++;
		}
		if (*s1 - *s2 == 0) atomicAdd(&(histo_private[5]), 1);

		i += stride;
		
	}

	__syncthreads();
	if (threadIdx.x < NUM_CLASSES) atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
}

int main() {
	const char* words[] = { "elephant", "lion", "zebra", "monkey", "tiger", "leopard" };

	int n = 4;

	printf("Please enter number of entries in file: \n");
	//scanf("%d\n", &n);

	// allocate memory for input
	char** input = (char**)malloc(n * sizeof(char*));

	if (input == NULL) {
		printf("Failed to allocate memory");
		return 1;
	}

	// randomly select from words array to fill input n times
	srand(42);
	for (int i = 0; i < n; i++) {
		int index = rand() % NUM_CLASSES;
		input[i] = strdup(words[index]);
	}

	// because number of input is large, in order to check correctness of parallel algorithm,
	// we first calculate hostogram seral
	int count_words_without_parallel[NUM_CLASSES];
	memset(count_words_without_parallel, 0, sizeof(count_words_without_parallel));

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < NUM_CLASSES; j++) {
			if (strcmp(input[i], words[j]) == 0) {
				count_words_without_parallel[j]++;
				break;
			}
		}
	}

	printf("count of each word in serial:\n");
	print_array(count_words_without_parallel);

	int global_histo[NUM_CLASSES];
	for (int i = 0; i < NUM_CLASSES; i++) global_histo[i] = 0;

	compute_histogram(global_histo, input, n);

	print_array(global_histo);

	return EXIT_SUCCESS;

}

cudaError_t compute_histogram(int* histo, char** input, int n) {
	char** dev_input = 0;
	int* dev_histo = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_histo, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_input, n * sizeof(char*));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_input, input, n * sizeof(char*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	clock_t start = clock();

	compute_hist_kernel << <NUM_BLOCKS, NUM_THREADS >> > (dev_histo, dev_input, n);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}


	double elapsedtime = (double)(clock() - start) / CLOCKS_PER_SEC;

	printf("elapsed time: %f\n", elapsedtime);

	cudaStatus = cudaMemcpy(histo, dev_histo, n * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(dev_histo);
	cudaFree(dev_input);

	return cudaStatus;
}

void print_array(int* a) {
	int i;
	printf("[-] histogram: ");
	for (i = 0; i < NUM_CLASSES; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}
