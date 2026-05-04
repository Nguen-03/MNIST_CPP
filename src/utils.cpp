#include "utils.h"


double cal_output(Vector4D &kernel, Vector3D &input, int out, int in, int row, int col){
    double result = 0;

    int n = kernel.thirdD; //kernel size is the 3rd D of kernel
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result += kernel.at(out, in, i, j) * input.at(in, row + i, col + j);

    return result;
}

void max_pool(Vector3D &mask, 
            Vector3D &output, 
            Vector3D &input, 
            int input_size, 
            int row, int col, 
            int pool_size, 
            int stride){

    double Max = -1e9;
    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;
    int row_index = 0, col_index = 0;

    for (int i = start_row; i < start_row + pool_size; ++i)

        for (int j = start_col; j < start_col + pool_size; ++j){
            mask.at(input_size, i, j) = 0;

            if (input.at(input_size, i, j) > Max){
                Max = input.at(input_size, i, j);
                row_index = i;
                col_index = j;
            }//end if
        }//end j

    mask.at(input_size, row_index, col_index) = 1;
    output.at(input_size, row, col) = Max;
}

void backward_max_pool(Vector3D &mask,
                     Vector3D &dInput, 
                     Vector3D &grads, 
                     int size, 
                     int row, 
                     int col, 
                     int pool_size, 
                     int stride){

    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;

    for (int i = start_row; i < start_row + pool_size; ++i)
        for (int j = start_col; j < start_col + pool_size; ++j){
            
            if(mask.at(size, i, j))
                dInput.at(size, i, j) += grads.at(size, row, col);
        }
}

void backward_Conv2d(Vector4D &kernels, 
                    Vector3D &input, 
                    Vector3D &grads, 
                    Vector3D &dInput, 
                    Vector4D &dKernels,
                    int out,
                    int in,
                    int row, 
                    int col, 
                    int kernel_size, 
                    int stride){

    int start_row, start_col;
    start_row = row * stride;
    start_col = col * stride;
    
    for (int i = 0; i < kernel_size; ++i)

        for (int j = 0; j < kernel_size; ++j){
            dKernels.at(out, in, i, j) += input.at(in, start_row + i, start_col + j) * grads.at(out, row, col);
            dInput.at(in, start_row + i, start_col + j) += kernels.at(out, in, i, j) * grads.at(out, row, col);
        }
}

void read_binary_images(string path, vector<pair<int, Vector3D>> &data, int number_of_images, int rows, int cols){
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
					data[i].second.at(0, r, c) = (double)images[index++] / 255.0;
				}
			}
		}
		file.close();
	}
}

void print_image(Vector3D& img) {
    for (int i = 0; i < 50; ++i)
        cout << "=";
    cout << endl;

    for (int r = 0; r < 28; ++r)
    {
        for (int c = 0; c < 28; ++c)
        {
            // Chuyển từ 0.0-1.0 về 0-255
            int pixel = static_cast<int>(img.at(0,r, c) * 255.0);

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

void shuffle_data(vector<pair<int, Vector3D>> &data){
    mt19937 gen(random_device{}());
    shuffle(data.begin(), data.end(), gen);
}