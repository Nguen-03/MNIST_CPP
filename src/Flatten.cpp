#include "Flatten.h"

Vector3D Flatten::forward(Vector3D &input){
    
    this->channels = input.firstD;
    this->rows = input.secondD;
    this->cols = input.thirdD;
    Vector3D output = Vector3D(1, 1, this->channels * this->rows * this->cols);
    // int index = 0;
    output.data = input.data;
    return output;
}

Vector3D Flatten::backward(Vector3D &grads){

    Vector3D output = Vector3D(this->channels, this->rows, this->cols);
    // int index = 0;

    output.data = grads.data;
    return output;
}

void Flatten::update(double lr){}

void Flatten::zero_grads(){}

void Flatten::save(ofstream &file){}

void Flatten::load(ifstream &file){}