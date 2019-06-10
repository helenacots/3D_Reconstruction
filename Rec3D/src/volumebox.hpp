//
//  volumebox.hpp
//  Rec3D
//
//  Created by Helena on 13/02/2019.
//  Copyright © 2019 Helena. All rights reserved.
//

#ifndef volumebox_hpp
#define volumebox_hpp

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include <sstream>
#include <getopt.h>
#include <vector>
#include <omp.h>
#include <math.h>

//opencv
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class VolumeBox {
	
private:
	void Update();
//    static bool compare(const Mat& C1, const Mat& C2);
	
public:
	
	//string filename;
	Mat coord;
	float ox, xmax, oy, ymax, oz, zmax;
	float voxelSize;
	int nx, ny, nz;
	int nvoxels;
	~VolumeBox();
	VolumeBox(int Ox, int Oy, int Oz, int Nx, int Ny, int Nz, float voxelsize);
	Mat computeVoxelProjections(vector<Mat> mP, vector< vector<Mat> > imagesC, vector< vector<Mat> > silC,int a, int b);
	void toVTK(const char* filename, Mat matrix);
	void toVTK_float(const char* filename, Mat matrix);
	vector<Mat> create_distance_lut(const vector<Mat> &centers, const vector<float> &idxs_goodvoxels);
	void sort_voxels_inRay_by_distance(vector< vector< vector< vector<Mat> > > > &lutsRay_aux, const vector< vector< vector< vector<Mat> > > > &voxel_luts, const vector<Mat> &distances, const int width, const int height, const vector<float> &idxs_goodvoxels);
    static bool compare(const Mat& C1, const Mat& C2);
    void store_depth_image(const char* filename, cv::Mat m, int npixels, int h, int w);
    void writeIsoSurfaceVTK(const char* filename, Mat TD,  vector<float> TD_array);
    static bool compareMax(float a, float b);

	//void Save();
	
};


#endif /* volumebox_hpp */
