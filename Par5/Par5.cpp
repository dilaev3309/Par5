#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#define GET_CUDA_STATUS(status) {gpuAssert((status), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "GPU assertion: %s %s %d\n", cudaGetErrorString(status), file, line);
        std::exit(status);
    }
}

#define GET_MPI_STATUS(status) {mpiAssert((status), __FILE__, __LINE__);}
inline void mpiAssert(int status, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI assertion: %s %s %d\n", status, file, line);
        std::exit(status);
    }
}

class parser {
public:
    parser(int argc, char** argv) {
        this->_grid_size = 512;
        this->_accur = 1e-6;
        this->_iters = 1000000;
        for (int i = 0; i < argc - 1; i++) {
            std::string arg = argv[i];
            if (arg == "-accur") {
                std::string dump = std::string(argv[i + 1]);
                this->_accur = std::stod(dump);
            }
            else if (arg == "-a") {
                this->_grid_size = std::stoi(argv[i + 1]);
            }
            else if (arg == "-i") {
                this->_iters = std::stoi(argv[i + 1]);
            }
        }

    };
    __host__ double accuracy() const {
        return this->_accur;
    }
    __host__ int iterations() const {
        return this->_iters;
    }
    __host__ int grid()const {
        return this->_grid_size;
    }
private:
    double _accur;
    int _grid_size;
    int _iters;

};

double corners[4] = { 10, 20, 30, 20 };

#define CALCULATE(A, B, size, i, j) \
    B[i * size + j] = 0.25 * (A[i * size + j - 1]\
                             + A[(i - 1) * size + j]\
                             + A[(i + 1) * size + j]\
                             + A[i * size + j + 1]);

__global__
void cross_calc(double* A, double* B, size_t size, size_t group_size) {
    //получаем индексы блоков и потоков
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    //вычисления
    if (!(i < 2 || j < 1 || j > size - 2 || i > group_size - 2)) {

        CALCULATE(A, B, size, i, j);

    }

}

__global__
void get_error_matrix(double* A_kernel, double* B_kernel, double* out, size_t size, size_t group_size) {
    //получаем инлекс
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    size_t idx = i * size + j;
    //ищем максимальную ошибку
    if (!(j == 0 || i == 0 || j == size - 1 || i == group_size - 1)) {

        out[idx] = std::abs(B_kernel[idx] - A_kernel[idx]);

    }

}

__global__
void bound_calc(double* A, double* B, size_t size, size_t group_size) {
    unsigned int up = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int down = blockIdx.x * blockDim.x + threadIdx.x;

    if (up == 0 || up > size - 2) return;

    if (up < size) {
        CALCULATE(A, B, size, 1, up);
        CALCULATE(A, B, size, (group_size - 2), down);
    }
}

double* A_kernel = nullptr,
* B_kernel = nullptr,
* dev_A = nullptr,
* dev_B = nullptr,
* dev_err = nullptr,
* dev_err_mat = nullptr,
* temp_stor = nullptr;

void free_mem() {
    if (dev_A) cudaFree(dev_A);
    if (dev_B) cudaFree(dev_B);
    if (dev_err_mat) cudaFree(dev_err_mat);
    if (temp_stor) cudaFree(temp_stor);
    if (A_kernel) cudaFree(A_kernel);
    if (B_kernel) cudaFree(B_kernel);
}

int near_power_two(size_t num) {
    int pow = 1;
    while (pow < num) {
        pow <<= 1;
    }
    return pow;
}


int main(int argc, char** argv) {

    auto exit_status = std::atexit(free_mem);

    if (exit_status != 0) {
        std::cout << "Register error" << std::endl;
        exit(-1);
    }

    parser input = parser(argc, argv);

    int size = input.grid();
    double min_error = input.accuracy();
    int max_iter = input.iterations();
    unsigned long full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);

    //Инициалмзация MPI
    int rank, group_size;
    GET_MPI_STATUS(MPI_Init(&argc, &argv));
    GET_MPI_STATUS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    GET_MPI_STATUS(MPI_Comm_size(MPI_COMM_WORLD, &group_size));

    //Устанавливаем номер девайса 
    int device_num = 0;
    cudaGetDeviceCount(&device_num);

    if (group_size > device_num || group_size < 1) {
        std::cout << "Invalid device number" << std::endl;
        std::exit(-1);
    }


    GET_CUDA_STATUS(cudaSetDevice(rank));


    size_t proc_area = size / group_size;
    size_t start_idx = proc_area * rank;

    //Инициализируем матрицы
    GET_CUDA_STATUS(cudaMallocHost(&A_kernel, sizeof(double) * full_size));
    GET_CUDA_STATUS(cudaMallocHost(&B_kernel, sizeof(double) * full_size));

    std::memset(A_kernel, 0, sizeof(double) * size * size);

    //Инициализируем границы
    A_kernel[0] = corners[0];
    A_kernel[size - 1] = corners[1];
    A_kernel[size * size - 1] = corners[2];
    A_kernel[size * (size - 1)] = corners[3];

    for (int i = 1; i < size - 1; i++) {
        A_kernel[i] = corners[0] + i * step;
        A_kernel[size * i] = corners[0] + i * step;
        A_kernel[(size - 1) + size * i] = corners[1] + i * step;
        A_kernel[size * (size - 1) + i] = corners[3] + i * step;
    }

    std::memcpy(B_kernel, A_kernel, sizeof(double) * full_size);

    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A_kernel[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // memory for one process
    if (rank != 0 && rank != group_size - 1) {
        proc_area += 2;
    }
    else {
        proc_area++;
    }

    size_t mem_size = size * proc_area;



    GET_CUDA_STATUS(cudaMalloc((void**)&dev_A, sizeof(double) * mem_size));
    GET_CUDA_STATUS(cudaMalloc((void**)&dev_B, sizeof(double) * mem_size));
    GET_CUDA_STATUS(cudaMalloc((void**)&dev_err, sizeof(double)));
    GET_CUDA_STATUS(cudaMalloc((void**)&dev_err_mat, sizeof(double) * mem_size));

    size_t offset = (rank != 0) ? size : 0;
    GET_CUDA_STATUS(cudaMemcpy(dev_A, A_kernel + (start_idx * size) - offset,
        sizeof(double) * mem_size, cudaMemcpyHostToDevice));
    GET_CUDA_STATUS(cudaMemcpy(dev_B, B_kernel + (start_idx * size) - offset,
        sizeof(double) * mem_size, cudaMemcpyHostToDevice));


    //Распределение временного хранилища
    size_t tmp_stor_size = 0;
    cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, size * proc_area);
    GET_CUDA_STATUS(cudaMalloc((void**)&temp_stor, tmp_stor_size));

    double* error;
    cudaMallocHost(&error, sizeof(double));
    *error = 1.0;

    cudaStream_t stream, mat_stream;
    GET_CUDA_STATUS(cudaStreamCreate(&stream));
    GET_CUDA_STATUS(cudaStreamCreate(&mat_stream));

    unsigned int threads_x = std::min(near_power_two(size), 1024);
    unsigned int blocks_y = proc_area;
    unsigned int blocks_x = size / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);

    int i = 0;
    //nvtxRangePushA("Main loop");
    //основной цикл
    while ((i < max_iter) && (*error) > min_error) {
        i++;
        //считаем итерации 
        bound_calc << <size, 1, 0, stream >> > (dev_A, dev_B, size, proc_area);

        cudaStreamSynchronize(stream);

        cross_calc << <gridDim, blockDim, 0, mat_stream >> > (dev_A, dev_B, size, proc_area);

        if (i % 100 == 0) {
            //получаем матрицу ошибок
            get_error_matrix << <gridDim, blockDim, 0, mat_stream >> > (dev_A, dev_B, dev_err_mat, size, proc_area);
            //находим максимальную ошибку
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, mem_size, mat_stream);

            GET_CUDA_STATUS(cudaStreamSynchronize(mat_stream));

            GET_MPI_STATUS(MPI_Allreduce(dev_err, dev_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

            //копируем в память хоста
            GET_CUDA_STATUS(cudaMemcpyAsync(error, dev_err, sizeof(double), cudaMemcpyDeviceToHost, mat_stream));

        }


        //Меняем границы
        //Верхняя граница
        if (rank != 0) {
            GET_MPI_STATUS(MPI_Sendrecv(dev_B + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0,
                dev_B + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        //Нижняя граница
        if (rank != group_size - 1) {
            GET_MPI_STATUS(MPI_Sendrecv(dev_B + (proc_area - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0,
                dev_B + (proc_area - 1) * size + 1,
                size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        cudaStreamSynchronize(mat_stream);

        //меняем матрицы
        std::swap(dev_A, dev_B);


    }

    //nvtxRangePop();

    if (rank == 0) {
        std::cout << "Error: " << *error << std::endl;
        std::cout << "Iteration: " << i << std::endl;
    }

    GET_MPI_STATUS(MPI_Finalize());
    return 0;
}