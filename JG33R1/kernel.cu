#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>

#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

#define BLOCK_SIZE 110

#define W 303
#define H 316

int picture[W][H];
int filtered_pic[W][H];

__device__ int dev_picture[W][H];
__device__ int dev_filtered_pic[W][H];



void fillTheArray(Mat src) {

    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            picture[i][j] = (int)src.at<uchar>(i, j);
}

void console(int array[W][H]) {

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++)
            std::cout << array[i][j] << " ";

        std::cout << std::endl;

    }
}

void filter() {

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {

            int window[9];
            window[0] = picture[i][j];


            if (i > 0 && picture[i - 1][j] >= 0)
                window[1] = picture[i - 1][j];
            else
                window[1] = 0;


            if (j > 0 && picture[i][j - 1] >= 0) 
                window[2] = picture[i][j - 1];
            else
                window[2] = 0;

            

            if (i > 0 && j > 0 && picture[i - 1][j - 1] >= 0) 
                window[3] = picture[i - 1][j - 1];
            else
                window[3] = 0;
            


            if (i < W - 1 && picture[i + 1][j] >= 0) 
                window[4] = picture[i + 1][j];
            else
                window[4] = 0;
            


            if (j < H - 1 && picture[i][j + 1] >= 0) 
                window[5] = picture[i][j + 1];
            else
                window[5] = 0;
            


            if (i < W - 1 && j < H - 1
                && picture[i + 1][j + 1] >= 0) 
                window[6] = picture[i + 1][j + 1];
            else
                window[6] = 0;
            


            if (i < W - 1 && j > 0
                && picture[i + 1][j - 1] >= 0) 
                window[7] = picture[i + 1][j - 1];
            else
                window[7] = 0;
            


            if (i > 0 && j < H - 1
                && picture[i - 1][j + 1] >= 0) 
                window[8] = picture[i - 1][j + 1];
            else
                window[8] = 0;
            
            int temp, x, y;
            for (x = 0; x < 9; x++) {
                temp = window[x];
                for (y = x - 1; y >= 0 && temp < window[y]; y--) {
                    window[y + 1] = window[y];
                }
                window[y + 1] = temp;
            }

            filtered_pic[i][j] = window[4];
        }
    }
}


__global__ void filterWxH() {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i + j >= W * H)
        return;

    dev_filtered_pic[i][j] = 0;


    int window[9];
    window[0] = dev_picture[i][j];


    if (i > 0)
        window[1] = dev_picture[i - 1][j];
    else
        window[1] = 0;


    if (j > 0)
        window[2] = dev_picture[i][j - 1];
    else
        window[2] = 0;



    if (i > 0 && j > 0)
        window[3] = dev_picture[i - 1][j - 1];
    else
        window[3] = 0;



    if (i < W - 1)
        window[4] = dev_picture[i + 1][j];
    else
        window[4] = 0;



    if (j < H - 1)
        window[5] = dev_picture[i][j + 1];
    else
        window[5] = 0;



    if (i < W - 1 && j < H - 1)
        window[6] = dev_picture[i + 1][j + 1];
    else
        window[6] = 0;



    if (i < W - 1 && j > 0)
        window[7] = dev_picture[i + 1][j - 1];
    else
        window[7] = 0;



    if (i > 0 && j < H - 1)
        window[8] = dev_picture[i - 1][j + 1];
    else
        window[8] = 0;

    int temp, x, y;
    for (x = 0; x < 9; x++) {
        temp = window[x];
        for (y = x - 1; y >= 0 && temp < window[y]; y--) {
            window[y + 1] = window[y];
        }
        window[y + 1] = temp;
    }

    dev_filtered_pic[i][j] = window[4];
}




int main() {

    Mat src = imread("phone.png", IMREAD_GRAYSCALE);
    Mat dst_CPU = src.clone();
    Mat dst_GPU = src.clone();

    fillTheArray(src);


#pragma region CPU

    auto start1 = high_resolution_clock::now();

    filter();
 
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);

    std::cout << "\nMedian Filter [CPU]: " << std::endl;
    cout << duration1.count() << " microseconds" << endl;

    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            dst_CPU.at<uchar>(i, j) = filtered_pic[i][j];

#pragma endregion

#pragma region GPU

    int block_count = ((W * H) - 1) / BLOCK_SIZE + 1;



    auto start2 = high_resolution_clock::now();
    cudaMemcpy(dev_picture, picture, W * H * sizeof(int), cudaMemcpyHostToDevice);

    filterWxH << <block_count, dim3(W, H) >> > ();

  

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);

    cudaMemcpy(picture, dev_picture, W * H * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
            dst_GPU.at<uchar>(i, j) = filtered_pic[i][j];

    std::cout << "\nMedian Filter [GPU]: " << std::endl;
    cout << duration2.count() << " microseconds" << endl;

#pragma endregion

    namedWindow("final_GPU");
    imshow("final_GPU", dst_GPU);

    namedWindow("final_CPU");
    imshow("final_CPU", dst_CPU);

    namedWindow("initial");
    imshow("initial", src);

    waitKey();
}