//
//  camera.hpp
//  Rec3D
//
//  Created by Helena on 13/02/2019.
//  Copyright © 2019 Helena. All rights reserved.
//

#ifndef camera_hpp
#define camera_hpp

#include <stdio.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <omp.h>
#include <math.h>
#include <getopt.h>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

class Camera{
	
private:
	void Update(); // function to update camera center
	
public:
	String filename;    // Name of xml file containing camera info
	cv::Mat P;              // Projection matrix
	cv::Mat K;              // Instrinsics matrix / Calibration Matrix
	cv::Mat R,t;            // Rotation matrix, translation vector
	cv::Mat center;         // Optical center of the camera
	cv::Mat orientation;    // Orientation of the camera
	int index;          // Index of the current camera in the set of N views
	vector<Mat> centers; // to store the centers of the cameras
	
	//Define methods:
	Camera();
	Camera(String cameraFile, int camId); // Constructor
	//void computeProjections(vector<Mat> &mP);
//	vector<Mat> computeCameraCenters(vector<Mat> matricesP,vector<Camera> cameraFiles);
	
};

#endif /* camera_hpp */
