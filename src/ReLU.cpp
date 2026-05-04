#include "ReLU.h"

ReLU::ReLU(){}
ReLU::~ReLU(){}
Vector3D ReLU::forward(Vector3D &input){

    this->input_cache = input;
    int input_size = input.firstD;
    int rows, cols;
    rows = input.secondD;
    cols = input.thirdD;
    Vector3D output = Vector3D(input_size, rows, cols);

    #pragma omp parallel for collapse(3)
    for (int size = 0; size < input_size; ++size)
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                output.at(size, row, col) = max(input.at(size, row, col), 0.0);
    return output;
}

Vector3D ReLU::backward(Vector3D &grads){
    Vector3D output = grads;
    int grads_size, rows, cols;
    grads_size = grads.firstD;
    rows = grads.secondD;
    cols = grads.thirdD;

    #pragma omp parallel for
    for (int i = 0; i < grads_size; ++i)
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                if (this->input_cache.at(i, row, col) <= 0)
                    output.at(i, row, col) = 0;
    return output;
}

void ReLU::update(double lr){}

void ReLU::zero_grads(){}

void ReLU::save(ofstream &file){}

void ReLU::load(ifstream &file){}