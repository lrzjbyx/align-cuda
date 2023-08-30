#include <cuda_runtime.h>
#include<iostream>
#include<cmath>
#include<vector>
#include<opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <device_launch_parameters.h>
using json = nlohmann::json;


#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}



__global__ void line_align_cuda(float * map_x_array,float * map_y_array,float * hh,float * ll,float x0,float y0,float t,int w,int h)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j < h && i < w) {
        float tx = x0 - hh[j] * cosf(M_PI/2 - (t * M_PI / 180.0));
        float ty = y0 + hh[j] * sinf(M_PI/2 - (t * M_PI / 180.0));
        
        float x1 = x0 + ll[i]* cosf(t * M_PI / 180.0);
        float y1 = y0 + ll[i]* sinf(t * M_PI / 180.0);

        map_x_array[j*w+i] = x1;
        map_y_array[j*w+i] = y1;
    }
}




static double radians(double t) {
    return t * M_PI / 180.0;
}


static  std::pair<float,float> oval(float h,float k, float a,float t,float c,float b){
    float x = h + a * cos(t) * cos(c) + b * sin(t) * sin(c);
    float y = k + a * cos(t) * sin(c) - b * sin(t) * cos(c);

    return std::make_pair(x,y);
}



static int elliptical_arc_length( float a, float  b, float  theta1, float  theta2){
    float h = std::pow(((a - b) / (a + b)), 2);
    float L = M_PI * (a + b) * (1 + ((3 * h) / (10 + sqrt(4 - 3 * h))));
    return L * abs(theta2 - theta1) / (2 * M_PI);
}

static std::vector<float> linspace(float start, float end, int num){

    std::vector<float> points(num);
    float step = static_cast<float>(end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        points[i] = start + i * step;
    }

    return points;
}







cv::Mat arc_align(cv::Mat image, json item){
    float ro = radians(item["rotation"].get<float>());
    float start_angle = radians(item["startAngle"].get<float>() / 16);
    float span_angle = radians(item["spanAngle"].get<float>()/16 );

    int width = elliptical_arc_length(item["a"].get<float>(),item["b"].get<float>(),start_angle,start_angle + span_angle);
    int height = item["h"];
    // 坐标圆点
    std::vector<float> cc = {item["rect"][2].get<float>() / 2 + item["x"].get<float>() + 
    item["rect"][0].get<float>(),item["rect"][3].get<float>() / 2 + +item["y"].get<float>() + item["rect"][1].get<float>()};
    std::vector<float> xx = linspace(start_angle, start_angle + span_angle,  width);
    std::vector<float> aa = linspace(item["a"].get<float>() - (item["h"].get<float>() / 2),item["a"].get<float>() + (item["h"].get<float>() / 2),  height);
    std::reverse(aa.begin(), aa.end());
    std::vector<float> bb = linspace(item["b"].get<float>() - (item["h"].get<float>() / 2), item["b"].get<float>() + (item["h"].get<float>() / 2),  height);
    std::reverse(bb.begin(), bb.end());

    // 创建 x 和 y 映射矩阵
    cv::Mat map_x, map_y;
    // cv::Mat dst( height,  width, CV_8UC1);;
    map_x.create(cv::Size( width,  height), CV_32FC1);
    map_y.create(cv::Size( width,  height), CV_32FC1);


    for(int j=0;j< height;j++){
        for(int i=0;i< width;i++){
            std::pair<float,float> xy = oval(cc[0],cc[1],aa[j],xx[i],ro,bb[j]);
            // int j = int(xy.second);
            // int i = int(xy.first);
            map_x.at<float>(j, i) = static_cast<float>(xy.first);
            map_y.at<float>(j, i) = static_cast<float>(xy.second);
            // map_y.at<float>(int(xy.second), int(xy.first)) = static_cast<float>(row);
        }
    }

    cv::Mat dst;
    cv::remap(image, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());


    cv::Mat result;
    cv::flip(dst, result ,1);
    return result;

}

cv::Mat line_align(cv::Mat image,json item){
    int height = item["h"];
    int width = item["l"];
    int l = item["l"];
    float x0 = item["rect"][2].get<float>() / 2 + item["x"].get<float>() + item["rect"][0].get<float>();
    float y0 = item["rect"][3].get<float>() / 2 + +item["y"].get<float>() + item["rect"][1].get<float>();
    int h = item["h"];
    float t = item["rotation"];
    std::vector<float> hh = linspace(-h / 2, h / 2, height);
    std::vector<float> ll = linspace(-l / 2, l / 2, width);
    std::reverse(ll.begin(), ll.end());


    int dev = 0;
    cudaSetDevice(dev);
    float* hh_array = hh.data();
    float* ll_array = ll.data();

    float* map_x_array = NULL;
    float* map_y_array = NULL;
    float* hh_dev = NULL;
    float* ll_dev = NULL;
    float* map_x_array_from_gpu = (float*)malloc(width*height);;
    float* map_y_array_from_gpu = (float*)malloc(width*height);;


    CHECK(cudaMalloc((void**)&hh_dev,hh.size()));
    CHECK(cudaMalloc((void**)&ll_dev,ll.size()));
    CHECK(cudaMalloc((void**)&map_x_array, width* height));
    CHECK(cudaMalloc((void**)&map_y_array, width* height));

    CHECK(cudaMemcpy(hh_dev,hh_array,hh.size(),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ll_dev,ll_array,ll.size(),cudaMemcpyHostToDevice));


    dim3 blockDim(16, 16); // 每个block的大小为16x16
    dim3 gridDim(( width + blockDim.x - 1) / blockDim.x, ( height + blockDim.y - 1) / blockDim.y); // 计算grid的大小

    line_align_cuda<<<gridDim, blockDim>>>(map_x_array,map_y_array,hh_dev,ll_dev,x0,y0,t, width,  height);


    CHECK(cudaMemcpy(map_x_array_from_gpu,map_x_array, width* height,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(map_y_array_from_gpu,map_y_array, width* height,cudaMemcpyDeviceToHost));


    cudaFree(hh_dev);
    cudaFree(ll_dev);
    free(map_x_array);
    free(map_y_array);
    free(hh_dev);
    free(ll_dev);
    cudaDeviceReset();




    // 创建 x 和 y 映射矩阵
    cv::Mat map_x, map_y;
    // cv::Mat dst( height,  width, CV_8UC1);;
    map_x.create(cv::Size(width, height), CV_32FC1);
    map_y.create(cv::Size(width, height), CV_32FC1);

    // map_x.create(cv::Size( width,  height), CV_32FC1,map_x_array_from_gpu);
    // map_y.create(cv::Size( width,  height), CV_32FC1,map_y_array_from_gpu);


    cv::Mat dst;
    cv::remap(image, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());


    cv::Mat result;
    cv::flip(dst, result ,1);
    
    return result;

}


cv::Mat run(cv::Mat image, json item){
    bool la = item["la"];
    bool mu = item["mu"];

    if(la == true){
        return arc_align(image,item);
    }else if (mu == true){
        return line_align(image,item);
    }

    return cv::Mat();

}
