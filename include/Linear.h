#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <vector>
#include "utils.h"
#include "Layers.h"
using namespace std;

class Linear : public Layers{
    public:
        int in_features, out_features;
        Vector3D input_cache, dInput;
        Vector2D W, dW;
        vector<double> b, db;
        Linear() {};
        Linear(int in_features, int out_features, bool bias = true);
        Vector3D forward(Vector3D &input) override;
        Vector3D backward(Vector3D &grads) override;
        void update(double lr) override;
        void zero_grads() override;
        void save(std::ofstream &file) override;
        void load(ifstream &file) override;
};

#endif