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
#include <boost/format.hpp>

#include<System.h>

#include <stdlib.h> 
using namespace std;
using namespace cv;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

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
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings depth_predict_model path_to_sequence start_index end_index" << endl;
        return 1;
    }

    vector<double> vTimestamps;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    cout<<"path_to_settings: " <<argv[1]<<endl;
    ORB_SLAM2::System SLAM(argv[1], argv[2], argv[3], ORB_SLAM2::System::STEREO,true);
    
    int  start_index = atoi( argv[5] );//读取序列的开端
    int  end_index = atoi( argv[6] );//序列的结束
    int  total_loop = 1;
    
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    
    //将图片裁剪成912x228, 1216x352
    int crop_width = 912; //x,y,width,height
    int crop_height = 228;
    
    // Main loop
    cv::Mat imLeft, imRight, depth_visual;
    //SLAM.LoadMap("../../pointcloud/test_map_09_26.bin");
    //getchar();
    
    for(int loop=0; loop<total_loop; loop++){
        for(int index=start_index; index<=end_index; index++)
        {
            // Read left and right images from file
	    //if(index>235){
	    //  sleep(20);
	    //}
            boost::format fmt ("%s/2011_09_30_drive_0033_sync/image_02/data/%010d.png");
	    imLeft = cv::imread( (fmt%argv[4]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
        
	    fmt = boost::format("%s/2011_09_30_drive_0033_sync/image_03/data/%010d.png");
	    imRight = cv::imread( (fmt%argv[4]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
	    
	    fmt = boost::format("%s/2011_09_26_drive_0014_sync/gt/%010d.png");
	    depth_visual = cv::imread( (fmt%argv[4]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
	    depth_visual = depth_read_gt(depth_visual);
	    
	    if(imLeft.empty() || imRight.empty())
            {
               continue;
            }
	
	    int u_coner = (imLeft.cols-crop_width)/2.0;
            int v_coner = imLeft.rows-crop_height;
	    Rect rect(u_coner,v_coner,crop_width,crop_height);
        
	    Mat imLeft_crop = imLeft(rect).clone();
	    Mat imRight_crop = imRight(rect).clone();
	    Mat depth_gt = depth_visual(rect).clone();

            // Pass the images to the SLAM system
            SLAM.TrackStereo(imLeft_crop,imRight_crop, depth_gt, index);     
        } 
    }
    
    // Stop all threads
    //SLAM.SaveMap("../../pointcloud/map.bin");
    
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}