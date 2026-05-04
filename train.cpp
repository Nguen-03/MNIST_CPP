#include "CNN.h"
using namespace std;

int main(){
	CNN *model = new CNN();
	
	vector<pair<int, Vector3D>> train, val;
	for (int i = 0; i < 3000; ++i){
		train.emplace_back(0, Vector3D(1, 28, 28));
	}

	for (int i = 0; i < 500; ++i)
		val.emplace_back(0, Vector3D(1,28, 28));
	
	read_binary_images("dataset/train.idx", train, 3000, 28, 28);
	read_binary_images("dataset/val.idx", val, 500, 28, 28);
	clock_t begin = clock();
	
	model->train(train, val, 16, 30, "lmao.bin");
	clock_t end = clock();
	double time_taken = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Run time: " << time_taken << " seconds" << endl;
	return 0;
}