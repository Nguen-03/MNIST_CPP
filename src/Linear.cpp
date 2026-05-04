#include "Linear.h"
#include <random>
Linear::Linear(int in_features, int out_features, bool bias){

    this->in_features = in_features;
    this->out_features = out_features;
    this->W = Vector2D(in_features, out_features);
    this->dW = Vector2D(in_features, out_features);
    this->b = vector<double>(out_features, 0.0);
    this->db = vector<double>(out_features, 0.0);
    this->dInput = Vector3D(1, 1, this->in_features);
    this->input_cache = Vector3D(1, 1, this->in_features);
    //initialize random follows normal distribution
    double scale = sqrt(2.0 / this->in_features);
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, scale);

    for (auto &x : this->W.data)
            x = dist(gen);
    
    if (bias){
        for (auto &x : b)
            x = dist(gen);
    }

}

Vector3D Linear::forward(Vector3D &input){
    Vector3D output = Vector3D(1, 1, this->out_features);
    this->input_cache.data = input.data;
    for (int i = 0; i < this->out_features; ++i){
        double sum = this->b[i];

        for (int j = 0; j < this->in_features; ++j)
            sum += input.at(0, 0, j) * this->W.at(j, i); //multiplicate the weight matrix (transpose) and input
        
        output.at(0, 0, i) = sum;
    }
    return output;
}

Vector3D Linear::backward(Vector3D &grads){

    if (!this->dInput.data.empty())
        fill(this->dInput.data.begin(), this->dInput.data.end(), 0.0);

    for (int i = 0; i < in_features; ++i)
        for (int j = 0; j < out_features; ++j)
            dInput.at(0, 0, i) += grads.at(0, 0, j) * W.at(i, j);

    for (int i = 0; i < in_features; ++i)
        for (int j = 0; j < out_features; ++j)
            dW.at(i, j) += input_cache.at(0, 0, i) * grads.at(0, 0, j);


    for (int j = 0; j < out_features; ++j)
        db[j] += grads.at(0, 0, j);

    return dInput;
}

void Linear::update(double lr){

    for (int in = 0; in < this->in_features; ++in)
        for (int out = 0; out < this->out_features; ++out){
            double grad = this->dW.at(in, out);
            this->W.at(in, out) -= lr * grad;
        }

    for (int out = 0; out < this->out_features; ++out){
        double grad = this->db[out];
        this->b[out] -= lr * grad;
    }
}

void Linear::zero_grads(){
    if (!this->dW.data.empty())
        fill(this->dW.data.begin(), this->dW.data.end(), 0.0);
    if (!this->db.empty())
        fill(this->db.begin(), this->db.end(), 0.0);
}


void Linear::save(ofstream &file){
    for (int i = 0; i < this->in_features; ++i) {
        for (int j = 0; j < this->out_features; ++j) {
            file.write((char *)&this->W.at(i, j), sizeof(double));
        }
    }

    for (int i = 0; i < this->out_features; ++i)
        file.write((char *)&this->b[i], sizeof(double));
}


void Linear::load(ifstream &file){
   
    for (int i = 0; i < this->in_features; ++i) {
        for (int j = 0; j < this->out_features; ++j) {
            file.read((char *)&this->W.at(i, j), sizeof(double));
        }
    }

    for (int i = 0; i < this->out_features; ++i)
        file.read((char *)&this->b[i], sizeof(double));
}