#include <cuda_runtime.h>
#include<iostream>
#include<cmath>
#include<vector>
#include<opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <device_launch_parameters.h>
#include<algorithm>
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

    

    // if(j < h && i < w && i==0) {
    if(j < h && i < w) {
        float tx = x0 - hh[j] * cosf(M_PI/2 - (t * M_PI / 180.0));
        float ty = y0 + hh[j] * sinf(M_PI/2 - (t * M_PI / 180.0));
        
        float x1 = tx + ll[i]* cosf(t * M_PI / 180.0);
        float y1 = ty + ll[i]* sinf(t * M_PI / 180.0);

        map_x_array[j*w+i] = x1;
        map_y_array[j*w+i] = y1;
        // printf("hh[%d]=%f\t\t----->i=%d\tj=%d\tw=%d\th=%d\tx1=%f\ty1=%f\ttx=%f\tty=%f\n",j,hh[j],i,j,w,h,x1,y1,tx,ty);
    }
}


__global__ void arc_align_cuda(float * map_x_array,float * map_y_array,float * xx,float *aa,float *bb,float x0,float y0,float ro,int w,int h){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    
    // if(j < h && i < w && i==0) {
    if(j < h && i < w) {
        float x1 = x0 + aa[j] * cosf(xx[i])*cosf(ro) + bb[j]* sinf(xx[i])*sinf(ro); 
        float y1 = y0 + aa[j] *cosf(xx[i])*sinf(ro) - bb[j]*sinf(xx[i])*cosf(ro);
        map_x_array[j*w+i] = x1;
        map_y_array[j*w+i] = y1;
    }
}



static double radians(double t) {
    return t * M_PI / 180.0;
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

    int dev = 0;
    cudaSetDevice(dev);

    float* xx_array = xx.data();
    float* aa_array = aa.data();
    float* bb_array = bb.data();


    float* xx_dev = NULL;
    float* aa_dev = NULL;
    float* bb_dev = NULL;


    float* map_x_array = NULL;
    float* map_y_array = NULL;

    float* map_x_array_from_gpu = (float*)malloc(width*height*sizeof(float));;
    float* map_y_array_from_gpu = (float*)malloc(width*height*sizeof(float));;

    CHECK(cudaMalloc((void**)&xx_dev,width*sizeof(float)));
    CHECK(cudaMalloc((void**)&aa_dev,height*sizeof(float)));
    CHECK(cudaMalloc((void**)&bb_dev,height*sizeof(float)));

    CHECK(cudaMalloc((void**)&map_x_array, width* height*sizeof(float)));
    CHECK(cudaMalloc((void**)&map_y_array, width* height*sizeof(float)));

    CHECK(cudaMemcpy(xx_dev,xx_array,  width*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(aa_dev,aa_array, height*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bb_dev,bb_array, height*sizeof(float),cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32); // 每个block的大小为16x16
    dim3 gridDim(( width + blockDim.x - 1) / blockDim.x, ( height + blockDim.y - 1) / blockDim.y); // 计算grid的大小

    float x0 = cc[0];
    float y0 = cc[1];

    arc_align_cuda<<<gridDim, blockDim>>>(map_x_array,map_y_array,xx_dev,aa_dev,bb_dev,x0,y0,ro,width, height);

    CHECK(cudaMemcpy(map_x_array_from_gpu,map_x_array, width* height*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(map_y_array_from_gpu,map_y_array, width* height*sizeof(float),cudaMemcpyDeviceToHost));

    cv::Mat map_x = cv::Mat(width*height, 1, CV_32F, map_x_array_from_gpu).reshape(0, height);
    cv::Mat map_y = cv::Mat(width*height, 1, CV_32F, map_y_array_from_gpu).reshape(0, height);

    cv::Mat dst;
    cv::remap(image, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());


    cv::Mat result;
    cv::flip(dst, result ,1);

    cudaFree(xx_dev);
    cudaFree(aa_dev);
    cudaFree(bb_dev);
    cudaFree(map_x_array);
    cudaFree(map_y_array);
    free(map_x_array_from_gpu);
    free(map_y_array_from_gpu);
    cudaDeviceReset();


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


    int dev = 0;
    cudaSetDevice(dev);
    // float* hh_array = (float*)malloc(height*sizeof(float));
    // float* ll_array = (float*)malloc(width*sizeof(float));
    // linspace(-l / 2, l / 2, width,ll_array);
    // linspace(-h / 2, h / 2, height,hh_array);

    std::vector<float> hh = linspace(-h / 2, h / 2, height);
    std::vector<float> ll = linspace(-l / 2, l / 2, width);
    std::reverse(ll.begin(), ll.end());

    float* hh_array = hh.data();
    float* ll_array = ll.data();

    float* map_x_array = NULL;
    float* map_y_array = NULL;
    float* hh_dev = NULL;
    float* ll_dev = NULL;
    float* map_x_array_from_gpu = (float*)malloc(width*height*sizeof(float));;
    float* map_y_array_from_gpu = (float*)malloc(width*height*sizeof(float));;


    CHECK(cudaMalloc((void**)&hh_dev,height*sizeof(float)));
    CHECK(cudaMalloc((void**)&ll_dev,width*sizeof(float)));
    CHECK(cudaMalloc((void**)&map_x_array, width* height*sizeof(float)));
    CHECK(cudaMalloc((void**)&map_y_array, width* height*sizeof(float)));

    CHECK(cudaMemcpy(hh_dev,hh_array,  height*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ll_dev,ll_array, width*sizeof(float),cudaMemcpyHostToDevice));


    dim3 blockDim(32, 32); // 每个block的大小为16x16
    dim3 gridDim(( width + blockDim.x - 1) / blockDim.x, ( height + blockDim.y - 1) / blockDim.y); // 计算grid的大小


    line_align_cuda<<<gridDim, blockDim>>>(map_x_array,map_y_array,hh_dev,ll_dev,x0,y0,t, width,  height);


    CHECK(cudaMemcpy(map_x_array_from_gpu,map_x_array, width* height*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(map_y_array_from_gpu,map_y_array, width* height*sizeof(float),cudaMemcpyDeviceToHost));

    cv::Mat map_x = cv::Mat(width*height, 1, CV_32F, map_x_array_from_gpu).reshape(0, height);
    cv::Mat map_y = cv::Mat(width*height, 1, CV_32F, map_y_array_from_gpu).reshape(0, height);


    cv::Mat dst;
    cv::remap(image, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    cudaFree(hh_dev);
    cudaFree(ll_dev);
    cudaFree(map_x_array);
    cudaFree(map_y_array);
    free(map_x_array_from_gpu);
    free(map_y_array_from_gpu);
    cudaDeviceReset();
    
    return dst;

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



#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;
using json = nlohmann::json;

cv::Mat align(cv::Mat mat, json jsonObject) {

    cv::Mat result = run(mat,jsonObject);

    return result;
}


PYBIND11_MODULE(sealnet_align, m) {
    m.def("align", [](py::array_t<unsigned char> input, py::dict json_data) {
        py::object json_module = py::module::import("json");
        py::str json_str = json_module.attr("dumps")(json_data);
        cv::Mat image = cv::Mat(input.shape(0), input.shape(1), CV_8UC3, input.mutable_data());
        json data = json::parse(json_str.cast<std::string>().c_str());
        cv::Mat result = align(image, data);
        std::vector<size_t> shape = { static_cast<size_t>(result.rows), static_cast<size_t>(result.cols), static_cast<size_t>(result.channels())};
        return py::array_t<unsigned char>(shape, result.data);
    });
}
