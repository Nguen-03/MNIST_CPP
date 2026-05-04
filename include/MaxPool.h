#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <iostream>
#include <vector>
#include "utils.h"
#include "Layers.h"
using namespace std;

class MaxPool : public Layers{
    public:
        int size, stride;
        Vector3D input_cache;
        Vector3D mask; // save indices have max values
        MaxPool() {};
        MaxPool(int size, int stride = 1);
        ~MaxPool() {};
        Vector3D forward(Vector3D &input) override;
        Vector3D backward(Vector3D &grads) override;
        void update(double lr) override;
        void zero_grads() override;
        void save(ofstream &file) override;
        void load(ifstream &file) override;
};

#endif