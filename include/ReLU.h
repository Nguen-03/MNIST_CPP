#ifndef RELU_H
#define RELU_H

#include <iostream>
#include <vector>
#include "utils.h"
#include "Layers.h"
using namespace std;


class ReLU : public Layers{
    public:
        Vector3D input_cache;
        ReLU();
        ~ReLU();
        Vector3D forward(Vector3D& input) override;
        Vector3D backward(Vector3D& grad) override;
        void update(double lr) override;
        void zero_grads() override;
        void save(ofstream &file) override;
        void load(ifstream &file) override;
};

#endif