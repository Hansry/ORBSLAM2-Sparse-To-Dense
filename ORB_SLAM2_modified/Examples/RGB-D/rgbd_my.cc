/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include <System.h>
#include <boost/format.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

using namespace std;
using namespace cv;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

//在kitti数据集上是用16位存储的
Mat depth_read(cv::Mat &depth, int sample_num){
    Mat depth_copy(depth.rows, depth.cols,CV_8UC1);
    Mat sample_depth(depth.rows, depth.cols,CV_8UC1);
    int nonZeroCount = 0;
    for(int row=0; row<depth.rows; row++){
      for(int col=0; col<depth.cols; col++){
	depth_copy.at<uchar>(row,col) = (uchar)floor(depth.at<int16_t>(row,col)/256.0);
	if(depth_copy.at<uchar>(row,col)>0.0){
	  nonZeroCount += 1;
	}
      }
    }
    float prob = (float)(sample_num)/(float)(nonZeroCount);
    for(int row=0;row<depth.rows;row++){
      for(int col=0;col<depth.cols;col++){
	sample_depth.at<uchar>(row,col) = 0;
	if(((double)rand())/RAND_MAX<prob){
	  sample_depth.at<uchar>(row,col) = depth_copy.at<uchar>(row,col);
	}
      }
    }
    return sample_depth;
}

Mat depth_read_gt(cv::Mat &depth){
    Mat depth_copy(depth.rows, depth.cols,CV_32FC1);
    for(int row=0; row<depth.rows; row++){
      for(int col=0; col<depth.cols; col++){
	depth_copy.at<float_t>(row,col) = (float_t)(depth.at<int16_t>(row,col)/256.0);
	}
      }
    return depth_copy;
}

int main(int argc, char **argv)
{
    if(argc != 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings depth_predict_model path_to_sequence start_index end_index " << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //argv[1]链接到ORBｖoc.txt
    //argv[2]包含各种内参
    //用argv[1]和argv[2]初始化SLAM系统
    ORB_SLAM2::System SLAM(argv[1],argv[2],argv[4],ORB_SLAM2::System::RGBD,true);

    int     start_index = atoi( argv[5] );//读取序列的开端
    int     end_index = atoi( argv[6] );//序列的结束
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    std::shared_ptr<torch::jit::script::Module> module;
    
    bool load_depth_model = false;
    bool use_predict_depth = false;
    
    if(load_depth_model){
       cout<<"load depth predict model, this could take a while !\n"<<endl;
       cout<<argv[4]<<endl;
       module = torch::jit::load(argv[4]);
    
       if(module!= nullptr){
          cout<<"\n load model ok !\n"<<endl;
       } 
       else{
           cout<<"\n load model failed !\n"<<endl;
	   assert(module != nullptr);
       }
    
       bool b_is_available = torch::cuda::is_available();
       if(b_is_available){
           cout<<"torch cuda is available !"<<endl;
       } 
    }
    
    //将图片裁剪成912x228, 1216x352
    int crop_width = 912; //x,y,width,height
    int crop_height = 228;
    // Main loop
    cv::Mat imRGB(crop_height, crop_width, CV_8UC3), imD(crop_height, crop_width, CV_8UC1);
    for(int loop=0; loop<1; loop++){
    for ( int index = start_index; index < end_index; index ++ )
    {
        //保证图片索引在10位左右
        boost::format fmt ("%s/rgb_2011_09_26_drive_0014_sync/%010d.png");
        imRGB = cv::imread( (fmt%argv[3]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
	if(imRGB.empty()){
	   cout<<"imRGB is empty !"<<endl;
	   break;
	}

        int u_coner = (imRGB.cols-crop_width)/2.0;
        int v_coner = imRGB.rows-crop_height;
	Rect rect(u_coner,v_coner,crop_width,crop_height);
	 
	Mat imRGB_crop = imRGB(rect).clone();
	Mat imD_crop;
	Mat imD_dense;
	int sample_num = 1000;
	
	if(module != nullptr && use_predict_depth){
	    //对模型进行时间测试
	    /*
	    clock_t start, finish;
	    start = clock();
	    std::vector<torch::jit::IValue> test_inputs;
            test_inputs.push_back(torch::rand({1, 3, 352, 1216}));
	    test_inputs.push_back(torch::rand({1, 1, 352, 1216}));

            module->forward(test_inputs).toTensor();
            test_inputs.pop_back();
            finish = clock();
	    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	    cout<<"cost time per image: "<<duration<<" s!"<<endl;
	    */
	    
	    fmt = boost::format("%s/gt_depth/%010d.png");
	    Mat imD_Pre = cv::imread( (fmt%argv[3]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
	    imD = depth_read(imD_Pre,sample_num);
	    if(imD.empty()){
	      cout<<"Sample number depth is empty !"<<endl;
	      continue;
	    }
	    imD_crop = imD(rect).clone();
	  
	    std::vector<torch::jit::IValue> inputs;
	    Mat imRGB_crop_tensor;
	    imRGB_crop.convertTo(imRGB_crop_tensor, CV_32F,1.0);
	  
	    torch::Tensor img_tensor_rgb = torch::from_blob(imRGB_crop_tensor.data,{1,imRGB_crop.rows,imRGB_crop.cols,3}, at::kFloat);
	    img_tensor_rgb = img_tensor_rgb.permute({0,3,1,2});
	    inputs.push_back(img_tensor_rgb);  
	    //auto img_var = torch::autograd::make_variable(img_tensor_rgb, false);
	    //inputs.push_back(img_var);
	  
	    std::vector<torch::jit::IValue> inputs_depth;
	    imD_crop.convertTo(imD_crop, CV_32F, 1.0);
	    torch::Tensor img_tensor_depth = torch::from_blob(imD_crop.data,{1,imD_crop.rows, imD_crop.cols,1}, at::kFloat);
	    img_tensor_depth = img_tensor_depth.permute({0,3,1,2});
	    inputs.push_back(img_tensor_depth);  
	    //auto img_var_depth = torch::autograd::make_variable(img_tensor_depth,false);
	    //inputs.push_back(img_var_depth);
	  	  
	    clock_t start, finish;
	    start = clock();
	    torch::Tensor out_tensor = module->forward(inputs).toTensor();
	    finish = clock();
	    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
	    cout<<"cost time per image: "<<duration<<" s!"<<endl;
	    
	    //cout<<out_tensor.size(2)<<out_tensor.size(3)<<out_tensor.size(2)<<endl;
	    
	    Mat imD_tensor_Mat(out_tensor.size(2),out_tensor.size(3), CV_32FC1, out_tensor.data<float>());
	    //imD_tensor_Mat.convertTo(imD_tensor_Mat,CV_8UC1);
	    imD_dense = imD_tensor_Mat.clone();
	 
	    //fmt = boost::format("%s/depth_09_30_33/%010d.png");
            //imD = cv::imread( (fmt%argv[3]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
	}
	else{
	   fmt = boost::format("%s/gt_2011_09_26_drive_0014_sync/%010d.png");
           imD = cv::imread( (fmt%argv[3]%index).str(), 0);
	   imD = depth_read_gt(imD);
	   
	   
	   int u_coner_depth = (imD.cols-crop_width)/2.0;
           int v_coner_depth = imD.rows-crop_height;
	   Rect rect(u_coner_depth,v_coner_depth,crop_width,crop_height);
	   
	   imD_crop = imD(rect).clone(); 
	   imD_dense = imD_crop.clone();
	}
	
	if(!imRGB_crop.empty() && !imD_crop.empty())
	{
	  //读取图片输入到SLAM系统中
	   SLAM.TrackRGBD( imRGB_crop, imD_dense, index);
	}
	else{
	  cout<<"Image Not Found!!"<<endl;
	  continue;
	}
    }
    }
    // Stop all threads
    cv::waitKey(0);
    
    SLAM.Shutdown();


    return 0;
}