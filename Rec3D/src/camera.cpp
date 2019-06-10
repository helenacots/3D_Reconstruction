//
//  camera.cpp
//  Rec3D
//
//  Created by Helena on 13/02/2019.
//  Copyright © 2019 Helena. All rights reserved.
//

#include "camera.hpp"

using namespace cv;
using namespace std;


Camera::Camera(){
	//just to initialize
}

void Camera::Update(){
	
}

Camera::Camera(String cameraFile, int camId){
	filename = cameraFile;
	index = camId;
	
	// Initialize camera parameteres
	// In opencv : Mat::zeros(int rows, int cols, int type);
	P = Mat::zeros(3,4,CV_32FC1); // P = K[R|t]
	K = Mat::zeros(3,3,CV_32FC1);
	R = Mat::zeros(3,3,CV_32FC1);
	t = Mat::zeros(3,1,CV_32FC1);
	center = Mat::zeros(3,1,CV_32FC1);
	orientation = Mat::zeros(3,1,CV_32FC1);
}

//vector<Mat> Camera::computeCameraCenters(vector<Mat> matricesP, vector<Camera> cameraFiles){
//
//	// Compute the camera center   PC = 0!  SVD of P, C is the right singular vector corresponding to the smallest singular value
//	Mat S,U,VT;
//
//	for(int i = 0; i < cameraFiles.size(); i++){
//		P = matricesP[i];
//		//decomposeProjectionMatrix(P,K,R,t);
//		SVD::compute(P, S, U, VT, SVD::FULL_UV);
//		Mat V = VT.t(); //the transpose
//		// C is the last column of V
//		center = V(Range(0,3),Range(3,4))/V.at<float>(3,3);
//		centers.push_back(center);
//	}
//
//	return centers;
//}
