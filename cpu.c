#define _CRT_SECURE_NO_WARNINGS

#define NUM_THREADS 1
#define NUM_CLASSES 6
#define STR_MAX_LEN 10

const char* words[] = {"elephant", "lion", "zebra", "monkey", "tiger", "leopard"};

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <omp.h>



void omp_check();
void print_array(int *a);

int main(int argc, char *argv[]){
    int n;

    printf("Please enter number of entries in file: \n");
    scanf("%uld\n", &n);

    // allocate memory for input
    char** input = (char**) malloc(n * sizeof(char*));
    if (input == NULL){
        printf("Failed to allocate memory");
        return 1;
    }

    // randomly select from words array to fill input n times
    srand(42);
    for (int i=0; i<n; i++){
        int index = rand() % NUM_CLASSES;
        input[i] = strdup(words[index]);
    }

    // because number of input is large, in order to check correctness of parallel algorithm,
    // we first calculate hostogram seral
    int count_words_without_parallel[NUM_CLASSES];
    memset(count_words_without_parallel, 0, sizeof(count_words_without_parallel));
    
    double starttime = omp_get_wtime();
    for (int i=0; i<n; i++){
        for(int j=0; j<NUM_CLASSES; j++){
            if (strcmp(input[i], words[j]) == 0){
                count_words_without_parallel[j]++;
                break;
            }
        }
    }
    double elapsedtime = omp_get_wtime() - starttime;
    printf("elapsed time serial: %f\n", elapsedtime);
    printf("count of each word in serial:\n");
    print_array(count_words_without_parallel);
    
    // allocate global histogram
    int global_hist[NUM_CLASSES];
    for (int i=0; i<NUM_CLASSES; i++) global_hist[i] = 0;
	
    starttime = omp_get_wtime();
    #pragma omp parallel num_threads(NUM_THREADS)
    {

        int local_hist[NUM_CLASSES];
        for (int i=0; i<NUM_CLASSES; i++) local_hist[i]=0;

        #pragma omp for nowait
        for (int i=0; i<n; i++){
            if      (strcmp(input[i], "elephant") == 0) local_hist[0] += 1;
            else if (strcmp(input[i], "lion") == 0) local_hist[1] += 1;
            else if (strcmp(input[i], "zebra") == 0) local_hist[2] += 1;
            else if (strcmp(input[i], "monkey") == 0) local_hist[3] += 1;
            else if (strcmp(input[i], "tiger") == 0) local_hist[4] += 1;
            else if (strcmp(input[i], "leopard") == 0) local_hist[5] += 1;
        }

        #pragma omp critical
            for (int i=0; i< NUM_CLASSES; i++) global_hist[i] += local_hist[i];
    }

    elapsedtime = omp_get_wtime() - starttime;
    
    printf("elapsed time parallel: %f\n", elapsedtime);
    
    print_array(global_hist);
    

}

void print_array(int *a) {
	int i;
	printf("[-] histogram: ");
	for (i = 0; i < NUM_CLASSES; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void omp_check() {
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is off.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}
