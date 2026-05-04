#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include "math.h"
// #define vector1D vector<double>
// #define vector2D vector<vector<double>>
// #define vector3D vector<vector<vector<double>>>
// #define vector4D vector<vector<vector<vector<double>>>>
using namespace std;


class Vector2D{
    public:
        int firstD, secondD;
        vector<double> data;
        Vector2D(){};
        Vector2D(int firstD, int secondD){
            this->firstD = firstD;
            this->secondD = secondD;
            this->data = vector<double>(firstD * secondD);
        }

        double &at(int a, int b){
            return this->data[(size_t)a * secondD + b];
        }
};

class Vector3D{
    public:
        int firstD, secondD, thirdD;
        vector<double> data;
        Vector3D(){};
        Vector3D(int firstD, int secondD, int thirdD){
            this->firstD = firstD;
            this->secondD = secondD;
            this->thirdD = thirdD;
            this->data = vector<double>(firstD * secondD * thirdD);
        }

        double &at(int a, int b, int c){
            return this->data[(size_t)a * secondD * thirdD + b * thirdD + c];
        }
};

class Vector4D{
    public:
        int firstD, secondD, thirdD, fourthD;
        vector<double> data;
        Vector4D(){};
        Vector4D(int firstD, int secondD, int thirdD, int fourthD){
            this->firstD = firstD;
            this->secondD = secondD;
            this->thirdD = thirdD;
            this->fourthD = fourthD;
            this->data = vector<double>(firstD * secondD * thirdD * fourthD);
        }

        double &at(int a, int b, int c, int d){
            return this->data[(size_t)a * secondD * thirdD * fourthD + b * thirdD * fourthD + c * fourthD + d];
        }
};

template <typename T>
ostream &operator <<(ostream &os, vector<T> &data ){
    for (auto x : data)
        cout << x << " ";
    cout << endl;
    return os;
} 

double cal_output(Vector4D &, Vector3D &, int, int, int, int);  // caculate the result of 1 element in output matrix (Conv2D)
void max_pool(Vector3D &mask,
            Vector3D &output, 
            Vector3D &input,
            int input_size, 
            int row, int col, 
            int pool_size, 
            int stride); // caculate output matrix, max pool forward
            
void backward_max_pool(Vector3D &mask,
                        Vector3D &dInput, 
                        Vector3D &grads, 
                        int size, 
                        int row, 
                        int col, 
                        int pool_size, 
                        int stride); //update gradient, max pool backward

void backward_Conv2d(Vector4D &kernels,
                    Vector3D &input,
                    Vector3D &grads,
                    Vector3D &dInput,
                    Vector4D &dKernels,
                    int in,
                    int out,
                     int row,
                     int col,
                     int kernel_size,
                     int stride); //of course backward Conv2D ☝️🤓
void read_binary_images(string path, vector<pair<int, Vector3D>> &data, int number_of_images, int rows, int cols); //read images into vectors
void print_image(Vector3D &img);
void shuffle_data(vector<pair<int, Vector3D>> &data);
#endif