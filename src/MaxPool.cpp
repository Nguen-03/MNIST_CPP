#include "MaxPool.h"

MaxPool::MaxPool(int size, int stride){
    this->size = size;
    this->stride = stride;
}

Vector3D MaxPool::forward(Vector3D &input){
    this->input_cache = input;
    int input_size, rows, cols;
    input_size = input.firstD;
    rows = input.secondD;
    cols = input.thirdD;
    if (this->mask.data.empty())
        this->mask = Vector3D(input_size, rows, cols);
    else{
        fill(mask.data.begin(), mask.data.end(), 0.0);
    }
    int row_out, col_out;
    row_out = (rows - this->size) / stride + 1;
    col_out = (cols - this->size) / stride + 1;
    Vector3D output = Vector3D(input_size, row_out, col_out);

    #pragma omp parallel for
    for (int i = 0; i < input_size; ++i)
        for (int row = 0; row < row_out; ++row)
            for (int col = 0; col < col_out; ++col){
                max_pool(this->mask, output, input,i, row, col, this->size, this->stride);
            }
            
    return output;
}

Vector3D MaxPool::backward(Vector3D &grads){
    int mask_size, rows, cols;
    mask_size = this->mask.firstD;
    rows = this->mask.secondD;
    cols = this->mask.thirdD;
    int row_out, col_out;
    row_out = (rows - this->size) / stride + 1;
    col_out = (cols - this->size) / stride + 1;
    
    Vector3D dInput = Vector3D(mask_size, rows, cols);
    #pragma omp parallel for
    for (int i = 0; i < mask_size; ++i)
        for (int row = 0; row < row_out; ++row)
            for (int col = 0; col < col_out; ++col)
                backward_max_pool(mask, dInput, grads, i, row, col, this->size, this->stride);
    return dInput;
}

void MaxPool::update(double lr){}

void MaxPool::zero_grads(){}

void MaxPool::save(ofstream &file){}

void MaxPool::load(ifstream &file){}