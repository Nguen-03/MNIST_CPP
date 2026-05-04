CXX = g++
CXXFLAGS = -Ofast -march=native -funroll-loops -fopenmp -Wall -std=c++17 -Iinclude

# 1. Tìm tất cả file trong src (đây là các file logic chung, không có hàm main)
CORE_SRCS = $(wildcard src/*.cpp)
CORE_OBJS = $(patsubst src/%.cpp,build/%.o,$(CORE_SRCS))

# 2. Định nghĩa các mục tiêu
all: train predict

# Build train: chỉ link train.o với các file logic chung
train: train.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o train.exe $^

# Build predict: chỉ link predict.o với các file logic chung
predict: predict.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o predict.exe $^

build:
	mkdir build

# Quy tắc biên dịch file .cpp thành .o
build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
# Lệnh xóa cho Windows (nếu dùng Linux/macOS thì đổi thành rm -f)
	del /s /q src\*.o *.o *.exe