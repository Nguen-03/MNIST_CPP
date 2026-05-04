#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <algorithm>
#include <fstream>
using namespace std;
// #define vector1D vector<double>
// #define vector2D vector<vector<double>>
// #define vector3D vector<vector<vector<double>>>
// #define vector4D vector<vector<vector<vector<double>>>>


template <typename T>
ostream& operator << (ostream &os, const vector<T> &a){
	for (auto& temp : a){
		os << temp << " ";
	}
	return os;
}

template <typename T>
ostream& operator << (ostream &os, const vector<vector<T>> &a){
	for (auto& temp : a){
        cout << temp;
        os << endl;
	}
	return os;
}

vector2D Matrix(int rows, int cols){
    vector2D init(rows, vector<double>(cols, 0.0));
    return init;
}

vector3D init3DVector(int firstD, int secondD, int thirdD){
    vector3D init(firstD, vector2D(secondD, vector<double>(thirdD, 0.0)));
    return init;
}

vector4D init4DVector(int firstD, int secondD, int thirdD, int fourthD){
    vector4D init(firstD, vector3D(secondD, vector(thirdD, vector<double> (fourthD, 0.0))));
    return init;
}

vector2D matrix_multi(vector2D a, vector2D b){
    int n = a.size();
	int p = a[0].size();
	int m = b[0].size();
	vector2D temp(n, vector<double>(m, 0));
	
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < m; ++j){
			for (int k = 0; k < p; ++k){
				temp[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return temp;
}



double cal_output(vector2D &kernel, vector2D &input, int row, int col){
    int n;
    double result = 0;
    n = kernel.size();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result += kernel[i][j] * input[row + i][col + j];
    return result;
}

void max_pool(vector2D &mask, vector2D &output, vector2D &input, int row, int col, int pool_size, int stride){
    double Max = -1e9;
    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;
    int row_index = 0, col_index = 0;
    for (int i = start_row; i < start_row + pool_size; ++i)
        for (int j = start_col; j < start_col + pool_size; ++j){
            mask[i][j] = 0;
            if (input[i][j] > Max){
                Max = input[i][j];
                row_index = i;
                col_index = j;
            }
        }
    mask[row_index][col_index] = 1;
    output[row][col] = Max;
}

void backward_max_pool(vector2D &mask, vector2D &dInput, vector2D &grads, int row, int col, int pool_size, int stride){

    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;
    for (int i = start_row; i < start_row + pool_size; ++i)
        for (int j = start_col; j < start_col + pool_size; ++j){
            if(mask[i][j])
                dInput[i][j] += grads[row][col];
        }
}

void backward_Conv2d(vector2D &kernels, 
                    vector2D &input, 
                    vector2D &grads, 
                    vector2D& dInput, 
                    vector2D &dKernels, 
                    int row, 
                    int col, 
                    int kernel_size, 
                    int stride){

    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;
    for (int i = 0; i < kernel_size; ++i)
        for (int j = 0; j < kernel_size; ++j){
            dKernels[i][j] += input[i + start_row][j + start_col] * grads[row][col];
            dInput[i + start_row][j + start_col] += kernels[i][j] * grads[row][col];
        }
}

void read_binary_images(string path, vector<pair<int, vector3D>> &data, int number_of_images, int rows, int cols){
	ifstream file(path, ios::binary);
	if (file.is_open()){
		file.seekg(24); // skip 24 bytes (number of images, magic number, rows, cols);
		vector<unsigned char> images(number_of_images * rows * cols);
		
		for (int i = 0; i < number_of_images; ++i){
			unsigned char temp = 0;
			file.read((char *)&temp, 1);
			data[i].first = (int)temp;
			file.read((char*)images.data() + i * rows * cols, rows * cols);
		}

		int index = 0;
		for (int i = 0; i < number_of_images; ++i){
			for (int r = 0; r < rows; ++r){
				for (int c = 0; c < cols; ++c){
					data[i].second[0][r][c] = (double)images[index++] / 255.0;
				}
			}
		}
		file.close();
	}
}

void print_image(const vector<vector<double>>& img) {
    for (int i = 0; i < 50; ++i)
        cout << "=";
    cout << endl;

    for (int r = 0; r < 28; ++r)
    {
        for (int c = 0; c < 28; ++c)
        {
            // Chuyển từ 0.0-1.0 về 0-255
            int pixel = static_cast<int>(img[r][c] * 255.0);

            // Chọn ký tự dựa trên độ sáng
            if (pixel > 200)
                cout << "@@"; // Rất sáng
            else if (pixel > 150)
                cout << "##";
            else if (pixel > 100)
                cout << "++";
            else if (pixel > 50)
                cout << "..";
            else
                cout << "  "; // Đen
        }
        cout << endl;
    }
    
    for (int i = 0; i < 50; ++i)
        cout << "=";
    cout << endl;

}

void shuffle_data(vector<pair<int, vector3D>> &data){
    mt19937 gen(random_device{}());
    shuffle(data.begin(), data.end(), gen);
}


struct Mat{
    int a;
    Mat(int lmao) : a(lmao){};
    void print(){
        cout << this->a;
    }
};
int main(){
    vector4D lmao = init4DVector(50, 50, 50, 50);
    vector<double> hehe;
    hehe = vector1D(6250000);
    clock_t begin = clock();
    // for (int i = 0; i < 50; ++i)
    //     for (int j = 0; j < 50; ++j)
    //         for (int k = 0; k < 50; ++k)
    //             for (int g = 0; g < 50; ++g)
    //                 lmao[i][j][k][g] += 1;
    for (int i = 0; i < 6250000; ++i)
        hehe[i] += 1;
    clock_t end = clock();
	double time_taken = double(end - begin) / CLOCKS_PER_SEC;
    Mat bruh(1);
    bruh.print();
    cout << endl;
    cout << time_taken;
}