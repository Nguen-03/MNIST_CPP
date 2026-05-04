#include "Conv2D.h"
#include "utils.h"
#include <random>

Conv2D::Conv2D(int in_channel, int out_channel, int kernel_size,  int stride, int padding, bool bias){
    this->kernel_size = kernel_size;
    this->in_channel = in_channel;
    this->out_channel = out_channel;
    this->stride = stride;
    this->padding = padding;
    this->bias = vector<double>(this->out_channel, 0.0);
    this->kernels = Vector4D(this->out_channel, this->in_channel, this->kernel_size, this->kernel_size);
    this->dKernels = Vector4D(this->out_channel, this->in_channel, this->kernel_size, this->kernel_size);
    this->db = vector<double>(this->out_channel, 0.0);
    
    int fan_in = this->in_channel * this->kernel_size * this->kernel_size;
    double stddev =  sqrt(2.0 / fan_in);
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stddev);

    #pragma omp parallel for collapse(4)
    for (int out = 0; out < this->out_channel; ++out)
        for (int in = 0; in < this->in_channel; ++in)
            for (int i = 0; i < this->kernel_size; ++i)
                for (int j = 0; j < this->kernel_size; ++j)
                    this->kernels.at(out, in, i, j) = dist(gen); //generate random weights in kernels
     
    std::normal_distribution<double> rand_bias(0.0, 0.01);
    if (bias == true){
        for (int i = 0; i < this->out_channel; ++i)
            this->bias[i] = rand_bias(gen); //generate random biases
    } //end if
    // cout << kernels.data;
} //end function

Conv2D::~Conv2D(){}
Vector3D Conv2D::forward(Vector3D &input){
    
    int input_size = input.secondD;
    int out_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    if (this->output.data.empty())
        this->output = Vector3D(this->out_channel, out_size, out_size);
    else
        fill(this->output.data.begin(), this->output.data.end(), 0.0);



    #pragma omp parallel for collapse(3)
    for (int o = 0; o < this->out_channel; ++o){
        for (int r = 0; r < out_size; ++r){
            for (int c = 0; c < out_size; ++c){
                for (int in = 0; in < in_channel; ++in){

                    int start_row, start_col;
                    start_row = r * stride;
                    start_col = c * stride;
                    output.at(o, r, c) += cal_output(this->kernels, input, o, in, start_row, start_col) ; //caculate outputs
                    
                }//end in_channel
                output.at(o, r, c) += this->bias[o];
            }//end col
        }//end row
    }//end out_channel

    // save cache for backward
    this->input_cache = input;
    this->output_cache = this->output;
    return this->output;
}

Vector3D Conv2D::backward(Vector3D &grads){

    if (this->dInput.data.empty())
        this->dInput = Vector3D(input_cache.firstD, input_cache.secondD, input_cache.thirdD);
    else{
        fill(this->dInput.data.begin(), this->dInput.data.end(), 0.0);
    }

    int out_rows, out_cols;
    out_rows = grads.secondD;
    out_cols = grads.thirdD;

    #pragma omp parallel for collapse(3)
    for (int o = 0; o < this->out_channel; ++o){
        for (int r = 0; r < out_rows; ++r){
            for (int c = 0; c < out_cols; ++c){
                for (int in = 0; in < this->in_channel; ++in){
                    backward_Conv2d(this->kernels, this->input_cache, grads, dInput, dKernels, o, in, r, c, this->kernel_size, stride);
                }
                this->db[o] += grads.at(o, r, c);
            }
        }
        
    }
    return dInput;
}

void Conv2D::update(double lr){

    #pragma omp parallel for collapse(4)
    for (int out = 0; out < this->out_channel; ++out)
        for (int in = 0; in < this->in_channel; ++in)
            for (int row = 0; row < this->kernel_size; ++row)
                for (int col = 0;col < this->kernel_size; ++col){

                    double grad = dKernels.at(out, in, row, col);
                    this->kernels.at(out, in, row, col) -= lr * grad;
                    
                }//end cols


    for (int out = 0; out < this->out_channel; ++out){
        double grad = this->db[out];
        this->bias[out] -= lr * grad;
    }
}

void Conv2D::zero_grads(){

    if (!this->dKernels.data.empty())
        fill(this->dKernels.data.begin(), this->dKernels.data.end(), 0.0);

    if (!this->db.empty())
        fill(this->db.begin(), this->db.end(), 0.0);
}

void Conv2D::save(ofstream &file){

    for (int out = 0; out < this->out_channel; ++out)
        for (int in = 0; in < this->in_channel; ++in)
            for (int row = 0; row < this->kernel_size; ++row)
                for (int col = 0;col < this->kernel_size; ++col)
                    file.write((char *)&this->kernels.at(out, in, row, col), sizeof(double));
    
    for (int out = 0; out < this->out_channel; ++out)
        file.write((char *)&this->bias[out], sizeof(double));
}

void Conv2D::load(ifstream &file){
    
    for (int out = 0; out < this->out_channel; ++out)
        for (int in = 0; in < this->in_channel; ++in)
            for (int row = 0; row < this->kernel_size; ++row)
                for (int col = 0;col < this->kernel_size; ++col)
                    file.read((char *)&this->kernels.at(out, in, row, col), sizeof(double));
    
    for (int out = 0; out < this->out_channel; ++out)
        file.read((char *)&this->bias[out], sizeof(double));
    
}