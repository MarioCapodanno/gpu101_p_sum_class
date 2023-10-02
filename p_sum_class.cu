#include <stdio.h>
#include <sys/time.h> // stampare tempo esecuzione
#include <stdlib.h>   // malloc

#define DIM 204800

// Check API call
#define CHECK(call)                                            \
  {                                                            \
    const cudaError_t err = call;                              \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

// Check to kernel call
#define CHECK_KERNELCALL()                                     \
  .                                                            \
  {                                                            \
    const cudaError_t err = cudaGetLastError();                \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void p_sum_cpu(float *p_sum, float *input, int length)
{
  p_sum[0] = input[0];
  int i;
  for (i = 1; i < length; ++i)
  {
    p_sum[i] = p_sum[i - 1] + input[i - 1];
  }
}

__global__ void p_sum_gpu(float *p_sum, float *input, int length)
{
  p_sum[0] = input[0];
  int i;
  for (i = 1; i < length; ++i)
  {
    p_sum[i] = p_sum[i - 1] + input[i - 1];
  }
}

int main(int argc, char *argv[])
{

  int i;
  double start_cpu, end_cpu, start_gpu, end_gpu;

  srand(time(NULL));

  float *p_sum_sw = (float *)malloc(sizeof(float) * DIM);
  float *input_v = (float *)malloc(sizeof(float) * DIM);
  float *p_sum_hw = (float *)malloc(sizeof(float) * DIM);

  for (i = 0; i < DIM; i++)
  {
    input_v[i] = rand() % 100; // genera numeri da 0 a 99
  }

  float *input_d, *p_sum_d;

  CHECK(cudaMalloc(&input_d, sizeof(float) * DIM));
  CHECK(cudaMalloc(&p_sum_d, sizeof(float) * DIM));

  start_cpu = get_time();
  p_sum_cpu(p_sum_sw, input_v, DIM);
  end_cpu = get_time();

  CHECK(cudaMemcpy(input_d, input_v, sizeof(float) * DIM, cudaMemcpyHostToDevice));

  start_gpu = get_time();
  dim3 blockPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, 1);
  p_sum_gpu<<<blockPerGrid, threadsPerBlock>>>(p_sum_d, input_d, DIM);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());
  end_gpu = get_time();

  CHECK(cudaMemcpy(p_sum_hw, p_sum_d, sizeof(float) * DIM, cudaMemcpyDeviceToHost));

  if (p_sum_hw[DIM - 1] != p_sum_sw[DIM - 1])
  {
    fprintf(stderr, "ERRORE RISULTATO SBAGLIATO SU GPU1\n");
  }

  printf("GPU TIME: %lf, CPU TIME: %lf", end_gpu - start_gpu, end_cpu - start_cpu);
  CHECK(cudaFree(input_d));
  CHECK(cudaFree(p_sum_d));

  free(p_sum_sw);
  free(input_v);
  free(p_sum_hw);

  return 0;
}