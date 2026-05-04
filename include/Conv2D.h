#ifndef CONV2D_H
#define CONV2D_H

#include <iostream>
#include <vector>
#include "utils.h"
#include "Layers.h"
using namespace std;

class Conv2D : public Layers{
        public:
                int kernel_size;
                int in_channel;
                int out_channel;
                int stride;
                int padding;

                Vector4D kernels, dKernels;
                vector<double> bias, db;
                Vector3D input_cache, dInput;
                Vector3D output, output_cache;
                Conv2D(){};
                Conv2D(int in_channel, int out_channel, int kernel_size, int stride = 1, int padding = 0, bool bias  = true);
                ~Conv2D();
                Vector3D forward(Vector3D &input) override;
                Vector3D backward(Vector3D &grads) override;
                void update(double lr) override;
                void zero_grads() override;
                void save(std::ofstream &file) override;
                void load(ifstream &file) override;
};
#endif