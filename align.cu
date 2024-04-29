#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <device_launch_parameters.h>
using json = nlohmann::json;

#define GPU_BLOCK_THREADS  512

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



__global__ void line_align_cuda(uint8_t*  src,uint8_t*  dst,int src_width, int src_height, int dst_width, int dst_height,float x,float y,float h,float l,float ro,uint8_t const_value_st, int edge){
    
    
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx      = position % dst_width;  // i
    int dy      = position / dst_width;  // j

    float hh = - dst_height/2 + dy * h/dst_height;
    float ll = - dst_width/2 + dx * l/dst_width;

    float tx = x - hh * cosf(M_PI/2 - ro);
    float ty = y + hh * sinf(M_PI/2 - ro);

    // float src_x = tx + ll* cosf(ro * M_PI / 180.0);
    // float src_y = ty + ll* sinf(ro * M_PI / 180.0);
    
    // float tx = x + hh * cosf(ro);
    // float ty = y - hh * sinf(ro);
    
    float src_x = tx + ll* cosf(ro);
    float src_y = ty + ll* sinf(ro);
    float c0, c1, c2;


    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * (src_width * 3 ) + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * (src_width * 3 ) + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * (src_width * 3 ) + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * (src_width * 3 ) + x_high * 3;
        }

        // same to opencv
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }



    uint8_t* pdst_c0 = dst + (dy * dst_width + dx) * 3;
    uint8_t* pdst_c1 = pdst_c0 + 1;
    uint8_t* pdst_c2 = pdst_c1 + 1;

    *pdst_c0 = static_cast<uint8_t>(c0);
    *pdst_c1 = static_cast<uint8_t>(c1);
    *pdst_c2 = static_cast<uint8_t>(c2);



}




__global__ void arc_align_cuda(uint8_t*  src,uint8_t*  dst,int src_width, int src_height, int dst_width, int dst_height,float x,float y,float a,float b,float h,float ro,float st,float se,uint8_t const_value_st, int edge)
{

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx      = position % dst_width;  // i
    int dy      = position / dst_width;  // j

    //    std::vector<float> xx = linspace(start_angle, start_angle + span_angle,  width);
    //    每个位置对应的角度
    float xx = se - dx *(se-st) / dst_width;
    float aa = (a + h/2 ) - dy * h / dst_height;
    float bb = (b + h/2 ) - dy * h / dst_height;

    //std::pair<float,float> xy = oval(cc[0],cc[1],aa[j],xx[i],ro,bb[j]);   

    float src_x = x + aa * cosf(xx) * cosf(ro) + bb * sinf(xx) * sinf(ro);
    float src_y = y + aa * cosf(xx) * sinf(ro) - bb * sinf(xx) * cosf(ro);
    float c0, c1, c2;

    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * (src_width * 3 ) + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * (src_width * 3 ) + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * (src_width * 3 ) + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * (src_width * 3 ) + x_high * 3;
        }

        // same to opencv
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }



    uint8_t* pdst_c0 = dst + (dy * dst_width + dx) * 3;
    uint8_t* pdst_c1 = pdst_c0 + 1;
    uint8_t* pdst_c2 = pdst_c1 + 1;

    *pdst_c0 = static_cast<uint8_t>(c0);
    *pdst_c1 = static_cast<uint8_t>(c1);
    *pdst_c2 = static_cast<uint8_t>(c2);


}



static double radians(double t) {
    return t * M_PI / 180.0;
}




static int elliptical_arc_length( float a, float  b, float  theta1, float  theta2){
    float h = std::pow(((a - b) / (a + b)), 2);
    float L = M_PI * (a + b) * (1 + ((3 * h) / (10 + sqrt(4 - 3 * h))));
    return L * abs(theta2 - theta1) / (2 * M_PI);
}

static dim3 grid_dims(int numJobs) {
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

cv::Mat arc_align(cv::Mat image, json item){
    // float ro = radians(item["rotation"].get<float>());
    float start_angle = radians(item["startAngle"].get<float>() / 16);
    float span_angle = radians(item["spanAngle"].get<float>()/16 );

    int dst_width = elliptical_arc_length(item["a"].get<float>(),item["b"].get<float>(),start_angle,start_angle + span_angle);
    int dst_height = static_cast<int>(item["h"].get<float>()); 

    int jobs   = dst_width * dst_height;
    auto grid  = grid_dims(jobs);
    auto block = block_dims(jobs);

    uint8_t*  src = image.data;
    uint8_t*  dst = (uint8_t*)malloc(dst_width*dst_height*3);
    int src_width = image.cols;
    int src_height  =  image.rows;

    std::vector<float> cc = {item["rect"][2].get<float>() / 2 + item["x"].get<float>() + 
    item["rect"][0].get<float>(),item["rect"][3].get<float>() / 2 + item["y"].get<float>() + item["rect"][1].get<float>()};
    float x = cc[0];
    float y =cc[1];
    float a = item["a"].get<float>();
    float b = item["b"].get<float>();
    float h = item["h"].get<float>();
    float ro = radians(item["rotation"].get<float>());
    float st = start_angle;
    float se = st + span_angle;
    uint8_t const_value_st= 255; 
    int edge = jobs;

    int dev = 0;
    cudaSetDevice(dev);


    uint8_t*  gpu_src; 
    uint8_t*  gpu_dst;
    auto size_image = src_width* src_height * 3;
    auto dst_size_image  = dst_height* dst_width * 3;
    cudaMalloc(&gpu_src, size_image);
    cudaMalloc(&gpu_dst, dst_size_image);
    cudaMemcpyAsync(gpu_src, src, size_image, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpu_dst, dst, dst_size_image, cudaMemcpyHostToDevice);

    arc_align_cuda<<<grid, block>>>(gpu_src,gpu_dst,src_width,src_height,dst_width,dst_height,x,y,a,b, h, ro,st,se,const_value_st,edge);

    cudaMemcpyAsync(src, gpu_src, size_image, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(dst, gpu_dst, dst_size_image, cudaMemcpyDeviceToHost);
    

    cv::Mat output(dst_height, dst_width, CV_8UC3, (void*)dst);


    cudaFree(gpu_src);
    cudaFree(gpu_dst);

    // free(dst);
    cudaDeviceReset();

    cv::Mat bgrImage;
    cv::cvtColor(output, bgrImage, cv::COLOR_BGR2RGB);

    return output;
    
}

cv::Mat line_align(cv::Mat image,json item){

    float ro = radians(item["rotation"].get<float>());
    int dst_width  = static_cast<int>(item["l"].get<float>());
    int dst_height  = static_cast<int>(item["h"].get<float>());

    int jobs   = dst_width * dst_height;
    auto grid  = grid_dims(jobs);
    auto block = block_dims(jobs);

    uint8_t*  src = image.data;
    uint8_t*  dst = (uint8_t*)malloc(dst_width*dst_height*3);
    int src_width = image.cols;
    int src_height  =  image.rows;

    float x = item["rect"][2].get<float>() / 2 + item["x"].get<float>() + item["rect"][0].get<float>();
    float y = item["rect"][3].get<float>() / 2 + +item["y"].get<float>() + item["rect"][1].get<float>();

    float h = item["h"].get<float>();
    float l = item["l"].get<float>();


    // cv::Point2f center(x, y); // 中心点
    // cv::Size2f size(l, h); // 尺寸，宽度和高度


    // // 创建旋转矩形
    // cv::RotatedRect rotatedRect(center,size, item["rotation"].get<float>());

    // // 获取旋转矩形的四个顶点
    // cv::Point2f vertices[4];
    // rotatedRect.points(vertices);

    // // 绘制旋转矩形
    // for (int i = 0; i < 4; i++) {
    //     cv::line(image, vertices[i], vertices[(i+1) % 4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    // }


    // cv::imwrite("122.png",image);

    // cv::Point2f center2((image.cols - 1) / 2.0, (image.rows - 1) / 2.0);

    // cv::Mat rot = cv::getRotationMatrix2D(center2, item["rotation"].get<float>(), 1.0);

    // // 旋转图像
    // cv::Mat wst;
    // cv::warpAffine(image, wst, rot, image.size());

    // cv::imwrite("133.png",wst);   

    uint8_t const_value_st= 255; 
    int edge = jobs;

    int dev = 0;
    cudaSetDevice(dev);


    uint8_t*  gpu_src; 
    uint8_t*  gpu_dst;
    auto size_image = src_width* src_height * 3;
    auto dst_size_image  = dst_height* dst_width * 3;
    cudaMalloc(&gpu_src, size_image);
    cudaMalloc(&gpu_dst, dst_size_image);
    cudaMemcpyAsync(gpu_src, src, size_image, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpu_dst, dst, dst_size_image, cudaMemcpyHostToDevice);


    line_align_cuda<<<grid, block>>>(gpu_src,gpu_dst,src_width,src_height,dst_width,dst_height,x,y,h,l, ro,const_value_st,edge);

    cudaMemcpyAsync(src, gpu_src, size_image, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(dst, gpu_dst, dst_size_image, cudaMemcpyDeviceToHost);
    

    cv::Mat output(dst_height, dst_width, CV_8UC3, (void*)dst);

    cudaFree(gpu_src);
    cudaFree(gpu_dst);

    // free(dst);
    cudaDeviceReset();

    cv::Mat bgrImage;
    cv::cvtColor(output, bgrImage, cv::COLOR_BGR2RGB);


    return output;

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



// int main(){
//     cv::Mat src = cv::imread("/home/zzu/scene_seal/align/image.png");
//     // cv::Mat src = cv::imread("/home/zzu/scene_seal/align/t0085.png");

//     // Align align(600,100);

//     // std::string jsonStringLine = R"({
//     //             "la":true,
//     //             "mu":false,
//     //             "x": 11.64244186046517,
//     //             "y": 328.3168604651162,
//     //             "rect": [
//     //                 95.0,
//     //                 -219.0,
//     //                 584.0755813953488,
//     //                 584.0755813953488
//     //             ],
//     //             "rotation": 0,
//     //             "text": "郑州大学档案项目组测试专用章",
//     //             "sequence": "从左到右",
//     //             "type": 5,
//     //             "startAngle": 5152,
//     //             "spanAngle": 4112,
//     //             "a": 292.0377906976744,
//     //             "b": 292.0377906976744,
//     //             "h": 154
//     //         }
//     //     )";
    
//     std::string jsonStringLine = R"(
//         {
//                 "la":false,
//                 "mu":true,
//                 "x": 83.8255813953487,
//                 "y": 728.8168604651161,
//                 "rect": [
//                     154.0,
//                     -119.0,
//                     325.6133720930233,
//                     10.0
//                 ],
//                 "rotation": 0,
//                 "text": "测试专用章",
//                 "type": 2,
//                 "sequence": "从左到右",
//                 "l": 325.6133720930233,
//                 "h": 97
//     }
//         )";






//     std::cout<<jsonStringLine<<std::endl;
//     json jsonObject = json::parse(jsonStringLine);

//     // json jsonObject = json::parse(jsonString);

//     cv::Mat result = run(src,jsonObject);

//     cv::imwrite("111.png",result);

//     return 0;
// }



