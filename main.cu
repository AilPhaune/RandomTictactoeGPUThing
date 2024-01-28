#include <vector>
#include <iostream>
#include <cuda_runtime.h>

struct Board {
    bool turnCross;
    uint16_t x, o;
};

__host__ __device__ void initBoard(Board* board) {
    board->turnCross = true;
    board->x = 0;
    board->o = 0;
}

#define GAME_NOT_ENDED 0
#define TIE 1
#define WINNER_X 2
#define WINNER_O 3

#define WIN1 0b000000111
#define WIN2 0b000111000
#define WIN3 0b111000000
#define WIN4 0b100100100
#define WIN5 0b010010010
#define WIN6 0b001001001
#define WIN7 0b100010001
#define WIN8 0b001010100

#define IS_WIN(bb) ((bb & WIN1) == WIN1 || (bb & WIN2) == WIN2 || (bb & WIN3) == WIN3 || (bb & WIN4) == WIN4 || (bb & WIN5) == WIN5 || (bb & WIN6) == WIN6 || (bb & WIN7) == WIN7 || (bb & WIN8) == WIN8)

/* POSITION:
 *  8 7 6
 *  5 4 3
 *  2 1 0
*/
__host__ __device__ void makeMove(Board* board, int pos) {
    if(board->turnCross) {
        board->x |= (1 << pos);
    } else {
        board->o |= (1 << pos);
    }
    board->turnCross = !board->turnCross;
}

__host__ __device__ void unmakeMove(Board* board, int pos) {
    int mask = ~(1 << pos);
    board->o &= mask;
    board->x &= mask;
    board->turnCross = !board->turnCross;
}

__host__ __device__ bool canMakeMove(Board* board, int pos) {
    return ((board->x | board->o) & (1 << pos)) == 0;
}

__host__ __device__ int winner(Board* board) {
    if(IS_WIN(board->x)) return WINNER_X;
    if(IS_WIN(board->o)) return WINNER_O;
    if((board->x | board->o) == 0b111111111) return TIE;
    return GAME_NOT_ENDED;
}

__host__ __device__ int regular_count(Board* board) {
    if(winner(board) != GAME_NOT_ENDED) {
        return 1;
    }
    int count = 0;
    for(int pos = 0; pos < 9; pos++) {
        if(!canMakeMove(board, pos)) continue;
        makeMove(board, pos);
        count += regular_count(board);
        unmakeMove(board, pos);
    }
    return count;
}

__global__ void gpu_single_count(Board* input, int* output) {
    Board b = input[0];
    if(threadIdx.x == 0) output[0] = regular_count(&b);
}

void GPU_single_thread_count() {
    Board board;
    initBoard(&board);
    Board* input = new Board[1];
    input[0] = board;
    int* output = new int[1];

    Board* gpu_input;
    cudaMalloc((void**) &gpu_input, sizeof(Board));
    int* gpu_output;
    cudaMalloc((void**) &gpu_output, sizeof(int));

    cudaMemcpy(gpu_input, input, sizeof(Board), cudaMemcpyHostToDevice);
    gpu_single_count<<<1, 1>>>(gpu_input, gpu_output);
    cudaMemcpy(output, gpu_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);
    delete[] input;
    delete[] output;

    std::cout << "GPU Single thread: " << output[0] << std::endl;
}

void CPU_count() {
    Board board;
    initBoard(&board);
    int count = regular_count(&board);
    std::cout << "CPU: " << count << std::endl;
}

int prepare_multithread_GPU_data(int depth, Board* board, std::vector<Board> &array) {
    if(depth == 0) {
        return 0;
    }
    if(winner(board) != GAME_NOT_ENDED) {
        return 1;
    }
    int count = 0;
    for(int pos = 0; pos < 9; pos++) {
        if(!canMakeMove(board, pos)) continue;
        makeMove(board, pos);
        if(depth == 1) array.push_back(*board);
        count += prepare_multithread_GPU_data(depth - 1, board, array);
        unmakeMove(board, pos);
    }
    return count;
}

__global__ void GPU_multi_count(Board* input, int* output, int size) {
    int arrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(arrayIndex >= size) return;

    output[arrayIndex] = regular_count(&input[arrayIndex]);
}

void GPU_multi_thread_count(int depth) {
    Board board;
    initBoard(&board);

    std::vector<Board> input;
    int sum = prepare_multithread_GPU_data(depth, &board, input);
    const int input_size = input.size();

    int* output = new int[input_size];

    Board* gpu_input;
    int* gpu_output;
    cudaMalloc((void**) &gpu_input, sizeof(Board)*input_size);
    cudaMalloc((void**) &gpu_output, sizeof(int)*input_size);

    cudaMemcpy(gpu_input, input.data(), sizeof(Board)*input_size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(256);
    dim3 numBlocks(input_size / threadsPerBlock.x + 1);
    GPU_multi_count<<<numBlocks, threadsPerBlock>>>(gpu_input, gpu_output, input_size);
    cudaMemcpy(output, gpu_output, sizeof(int)*input_size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    for(int i = 0; i < input_size; i++) {
        sum += output[i];
    }
    delete[] output;

    std::cout << "GPU " << input_size << " threads (depth " << depth << "): " << sum << std::endl;
}

int main() {
    std::cout << "### TICTACTOE TERMINAL POSITION COUNT ###" << std::endl;
    std::cout << "Expected: 255168" << std::endl;

    CPU_count();
    GPU_single_thread_count();
    GPU_multi_thread_count(1);
    GPU_multi_thread_count(2);
    GPU_multi_thread_count(3);
    GPU_multi_thread_count(4);
    GPU_multi_thread_count(5);
    GPU_multi_thread_count(6);
    GPU_multi_thread_count(7);
    GPU_multi_thread_count(8);
}