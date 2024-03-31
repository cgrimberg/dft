#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

#define TOTALELEMENTS 2048//(64*1024)
#define SAMPLINGFREQ   1024
#define PI 			   3.14159

cuDoubleComplex iii = make_cuDoubleComplex(0, 1);

cuDoubleComplex * ListO;//[TOTALELEMENTS];
cuDoubleComplex * GPUListSrc;//[TOTALELEMENTS];
cuDoubleComplex * GPUListDst;//[TOTALELEMENTS];
cuDoubleComplex * ListD;//[TOTALELEMENTS];

struct pair
{
	cuDoubleComplex mag;
	double freq;
	double phase;
};

struct pair * signalList;
struct pair * GPUsignalList;

__host__ void printContents(cuDoubleComplex * a, int start, int count)
{
	for (int i = 0; i < count; i++)
	{
		printf("%f + %fi\t", a[start+i].x, a[start+i].y);
	}
	printf("\n");
	return;
}

__device__ __host__ cuDoubleComplex cuCpow(cuDoubleComplex a, int b)
{
	cuDoubleComplex o = make_cuDoubleComplex(1, 0);
	for (int i = 0; i < b; i++)
	{
		o = cuCmul(o, a);
	}
	return o;
}

__global__
void fft(cuDoubleComplex *src, cuDoubleComplex *dst, int length, cuDoubleComplex rou)
{
	unsigned int ThrPerBlk = blockDim.x;
	unsigned int MYbid = blockIdx.x;
	unsigned int MYtid = threadIdx.x;
	unsigned int MYgtid = ThrPerBlk * MYbid + MYtid;
	cuDoubleComplex wn1 = cuCpow(rou, MYgtid);
	cuDoubleComplex wnn = make_cuDoubleComplex(1,0);
	int i;
	for (i = 0; i < length; i++)
	{
		dst[MYgtid] = cuCadd(dst[MYgtid], cuCmul(src[i],wnn));
		wnn = cuCmul(wnn, wn1);
	}
}

__global__
void ifft(cuDoubleComplex *src, cuDoubleComplex *dst, int length, cuDoubleComplex rou)
{
	unsigned int ThrPerBlk = blockDim.x;
	unsigned int MYbid = blockIdx.x;
	unsigned int MYtid = threadIdx.x;
	unsigned int MYgtid = ThrPerBlk * MYbid + MYtid;
	cuDoubleComplex wn1 = cuCpow(rou, MYgtid);
	cuDoubleComplex wnn = make_cuDoubleComplex(1,0);

	int i;
	for (i = 0; i < length; i++)
	{
		dst[MYgtid] = cuCadd(dst[MYgtid], cuCdiv(src[i], wnn));
		wnn = cuCmul(wnn, wn1);
	}
	dst[MYgtid] = cuCdiv(dst[MYgtid],make_cuDoubleComplex(length,0));
}

__global__
void create(pair * signals, int length, int random, cuDoubleComplex * dst)
{
    int i;
	pair item;
	unsigned int ThrPerBlk = blockDim.x;
	unsigned int MYbid = blockIdx.x;
	unsigned int MYtid = threadIdx.x;
	unsigned int MYgtid = ThrPerBlk * MYbid + MYtid;
	float t = ((float)MYgtid)/SAMPLINGFREQ;
    for(i=0; i<length; i++)
    {
        item = signals[i];
		dst[MYgtid] = cuCadd(dst[MYgtid], cuCmul(item.mag, make_cuDoubleComplex((double)(cosf(item.freq*2*PI*t + item.phase)), (double)(sinf(item.freq*2*PI*t + item.phase)))));
    }
	//dst[MYgtid] = cuCadd(dst[MYgtid], make_cuDoubleComplex((rand() % random ) * ((rand() % 2)*2-1),0));
}

double mse(cuDoubleComplex * numb1, cuDoubleComplex * numb2, int elements)
{
    int i;
	double error = 0;
    for(i=0;i<elements;i++)
    {
        error += cuCabs(cuCpow(cuCsub(numb1[i],numb2[i]), 2));
    }
    return error/elements;

}

int main(int argc, char **argv)
{
	srand(time(NULL));


	ListO = (cuDoubleComplex *) malloc(TOTALELEMENTS*sizeof(cuDoubleComplex));
	if (ListO==NULL)
	{
		printf("Cannot allocate memory for NumebrsO.\n");
		exit(EXIT_FAILURE);
	}
    ListD = (cuDoubleComplex *) malloc(TOTALELEMENTS*sizeof(cuDoubleComplex));
	if (ListD==NULL)
	{
		free(ListO);
		printf("Cannot allocate memory for NumebrsQ.\n");
		exit(EXIT_FAILURE);
	}


	// max of 10 sinusoids
	signalList = (pair *) malloc(10*sizeof(pair));
	if (signalList==NULL)
	{
		free(ListD);
		free(ListO);
		printf("Cannot allocate memory for NumebrsQ.\n");
		exit(EXIT_FAILURE);
	}

	////////////////////////////// define sinusoids hre.
	int sinusoidCount = 5;
	signalList[0].mag = make_cuDoubleComplex(8, 0);
	signalList[0].freq = 0;
	signalList[0].phase = 0;
	
	signalList[1].mag = make_cuDoubleComplex(0,-3.5);
	signalList[1].freq = 50;
	signalList[1].phase = 0;

	signalList[2].mag = make_cuDoubleComplex(0,3.5);
	signalList[2].freq = -50;
	signalList[2].phase = 0;

	signalList[3].mag = make_cuDoubleComplex(0,-5);
	signalList[3].freq = 120;
	signalList[3].phase = 0;

	signalList[4].mag = make_cuDoubleComplex(0,5);
	signalList[4].freq = -120;
	signalList[4].phase = 0;
	int noiseMagnitude = 20;
	//printf("%f\n\n\n", signalList[1].mag*cosf(signalList[1].freq*2*PI*0.250 + signalList[1].phase));

	//float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t cudaStatus, cudaStatus2, cudaStatus3;
	//char InputFileName[255], OutputFileName[255], ProgName[255];
	int ThrPerBlk=256;
	int NumBlocks=TOTALELEMENTS/ThrPerBlk;
	cudaDeviceProp GPUprop;
	//unsigned long SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];

	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	// SupportedKBlocks = (unsigned int)GPUprop.maxGridSize[0] * (unsigned int)GPUprop.maxGridSize[1] * (unsigned int)GPUprop.maxGridSize[2] / 1024;
	// SupportedMBlocks = SupportedKBlocks / 1024;
	// sprintf(SupportedBlocks, "%lu %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	// MaxThrPerBlk = (unsigned int)GPUprop.maxThreadsPerBlock;


	// Allocate GPU buffer for the input and output images
	cudaStatus = cudaMalloc((void**)&GPUListSrc, TOTALELEMENTS*sizeof(cuDoubleComplex));
	cudaStatus2 = cudaMalloc((void**)&GPUListDst, TOTALELEMENTS*sizeof(cuDoubleComplex));
	cudaStatus3 = cudaMalloc((void**)&GPUsignalList, TOTALELEMENTS*sizeof(pair));
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess) || (cudaStatus3 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUsignalList, signalList, TOTALELEMENTS*sizeof(pair), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}


	create <<< NumBlocks, ThrPerBlk >>> (GPUsignalList, sinusoidCount, noiseMagnitude, GPUListSrc);

	printf("Finish create\n");
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy(ListO, GPUListSrc, TOTALELEMENTS*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}
	//printContents(ListO, 0, 20);

	cuDoubleComplex rou = make_cuDoubleComplex(cosf(-2*PI/TOTALELEMENTS), sinf(-2*PI/TOTALELEMENTS));
	fft <<< NumBlocks, ThrPerBlk >>> (GPUListSrc, GPUListDst, TOTALELEMENTS, rou);
	
	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(ListD);
		free(ListO);
		free(signalList);
		exit(EXIT_FAILURE);
	}

	printf("Finish fft\n");

/**************** delete later 
	cudaStatus = cudaMemcpy(ListO, GPUListDst, TOTALELEMENTS*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}
	printContents(ListO, 235, 10);
*****/

	ifft <<< NumBlocks, ThrPerBlk >>> (GPUListDst, GPUListSrc, TOTALELEMENTS, rou);

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(ListD);
		free(ListO);
		free(signalList);
		exit(EXIT_FAILURE);
	}
	printf("Finish ifft\n");
	cudaStatus = cudaMemcpy(ListD, GPUListSrc, TOTALELEMENTS*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(ListD);
		free(ListO);
		free(signalList);
		exit(EXIT_FAILURE);
	}

	printf("Mean Squared Error: %f\n", mse(ListO, ListD, TOTALELEMENTS));

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUListDst);
	cudaFree(GPUListSrc);
	cudaFree(GPUsignalList);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(ListD);
		free(ListO);
		free(signalList);
		exit(EXIT_FAILURE);
	}
	free(ListD);
	free(ListO);
	free(signalList);
	return(EXIT_SUCCESS);
}



