//
//  main.cpp
//  Rec3D
//
//  Created by Helena on 09/02/2019.
//  Copyright © 2019 Helena. All rights reserved.
//
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include <sstream>
#include <getopt.h>
#include <iterator>
#include <vector>
#include "omp.h"
#include <math.h>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#define CV_LOAD_IMAGE_ANYDEPTH 2
#define CV_LOAD_IMAGE_ANYCOLOR 4

#include "camera.hpp"
#include "tinyxml2.h"
#include "tinyxml2.cpp"
#include "volumebox.hpp"

#include <daisy/daisy.h>
#include <kutility/image.h>
#include <kutility/image_io_png.h>
//#include "image_io_png.cpp"
#include "png.h"

using namespace cv;
using namespace std;
using namespace tinyxml2;
using namespace kutility;


int main(int argc, const char * argv[]) {


    //    Initialize parameters
    
    String imagesDir, silhouettesDir, calibrationDir;
    
//    imagesDir = "/Users/helena/Documents/TFG/data/imagesDir";
//    silhouettesDir = "/Users/helena/Documents/TFG/data/silhouettesDir";
//    calibrationDir = "/Users/helena/Documents/TFG/data/calibration";
    imagesDir = "/Users/helena/Documents/TFG/data/imagesDir";
    silhouettesDir = "/Users/helena/Documents/TFG/data/silhouettesDir";
    calibrationDir = "/Users/helena/Documents/TFG/data/calibration";
    
    //define vectors where the names of the images and silhouettes are stored
    vector<String> pngFiles; // Vector to store the name of the input png images
    vector<String> pbmFiles; // Vector to store the name of the input pbm images (silhouettes)
    
    // FOR THE IMAGES
    vector<Mat> images; // Store png images
    vector< vector<Mat> > imagesC(8,vector<Mat>()); // Store the png separating from different cameras
    vector< vector<string> > imagesNames(8,vector<string>());     // Store the png images name
    
    // FOR THE SILHOUETTES
    vector<Mat> silhouettes;        // Store the pbm images.
    vector< vector<Mat> > silC(8,vector<Mat>());       // Store the pbm images separating from different cameras
    vector< vector<string> > silNames(8,vector<string>());        // Store the pbm images name
    
    vector<String> xmlFiles;         // Store the xml camera files
    
    String camNames[8];
    String camSilNames[8];     // Store sil cam names silh_camXX
    
    // We will use it to store all the images frames from every
   /* camNames[0] = imagesDir + ("/cam00");
    camNames[1] = imagesDir + ("/cam01");
    camNames[2] = imagesDir + ("/cam02");
    camNames[3] = imagesDir + ("/cam03");
    camNames[4] = imagesDir + ("/cam04");
    camNames[5] = imagesDir + ("/cam05");
    camNames[6] = imagesDir + ("/cam06");
    camNames[7] = imagesDir + ("/cam07");
    
    // We will use it to store all the silhouettes frames from ever camera
    camSilNames[0] = silhouettesDir + "/silh_cam00";
    camSilNames[1] = silhouettesDir + "/silh_cam01";
    camSilNames[2] = silhouettesDir + "/silh_cam02";
    camSilNames[3] = silhouettesDir + "/silh_cam03";
    camSilNames[4] = silhouettesDir + "/silh_cam04";
    camSilNames[5] = silhouettesDir + "/silh_cam05";
    camSilNames[6] = silhouettesDir + "/silh_cam06";
    camSilNames[7] = silhouettesDir + "/silh_cam07";
	*/
    
    // We will use it to store images from each of the cameras
    camNames[0] = imagesDir + ("/cam00_00023_0000008550");
    camNames[1] = imagesDir + ("/cam01_00023_0000008550");
    camNames[2] = imagesDir + ("/cam02_00023_0000008550");
    camNames[3] = imagesDir + ("/cam03_00023_0000008550");
    camNames[4] = imagesDir + ("/cam04_00023_0000008550");
    camNames[5] = imagesDir + ("/cam05_00023_0000008550");
    camNames[6] = imagesDir + ("/cam06_00023_0000008550");
    camNames[7] = imagesDir + ("/cam07_00023_0000008550");
    
    // We will use it to store silhouettes from each of the cameras
    camSilNames[0] = silhouettesDir + "/silh_cam00_00023_0000008550";
    camSilNames[1] = silhouettesDir + "/silh_cam01_00023_0000008550";
    camSilNames[2] = silhouettesDir + "/silh_cam02_00023_0000008550";
    camSilNames[3] = silhouettesDir + "/silh_cam03_00023_0000008550";
    camSilNames[4] = silhouettesDir + "/silh_cam04_00023_0000008550";
    camSilNames[5] = silhouettesDir + "/silh_cam05_00023_0000008550";
    camSilNames[6] = silhouettesDir + "/silh_cam06_00023_0000008550";
    camSilNames[7] = silhouettesDir + "/silh_cam07_00023_0000008550";
    
    
    int64 e1 = getTickCount();
    
    /* --------------------------------- READ THE DATASET ---------------------------------
     
     - Read the images and the silhouettes from the dataset.
     - Store them classifying them by cameras.
     - Read xml files for every camera and save matrices P.
     - Calculate camera parameters needed. */

    // READ IMAGES PNG DIRECTLY FROM THE CAMERAS
    int h,w; //height and width of the images

    glob(imagesDir.operator+=("/*.png"),pngFiles,false); //Get all the png files in the input images directory
    Mat imageNow;
    Mat image_raw;
    string currentName;
    if (pngFiles.size() == 0){
        cerr << "\nERROR: No png images were found.";
        return EXIT_FAILURE;
    }else{
        // We are reading the png files directly taken from different cameras.
        int x0 = 0,x1 = 0,x2 = 0,x3 = 0,x4 = 0,x5 = 0,x6 = 0,x7 = 0;
        // In case we want to read all the frames, one has to uncomment the xN++.
        for(int i = 0; i < pngFiles.size(); i++){
            imageNow = imread(pngFiles[i], CV_LOAD_IMAGE_ANYCOLOR); //dims = 2, data = uchar
            imageNow.convertTo(imageNow, CV_32FC3); // fem la conversio pq els valors de la imatge siguin FLOAT de 0 - 255
            h = imageNow.rows;
            w = imageNow.cols;
            currentName = pngFiles[i];
            if(pngFiles[i].find(camNames[0])==0){
                imagesC[0].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[0][x0]);
                //x0++;
                imagesNames[0].push_back(currentName);
            }else if(pngFiles[i].find(camNames[1])==0){
                imagesC[1].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[1][x1]);
                //x1++;
                imagesNames[1].push_back(currentName);
            }else if(pngFiles[i].find(camNames[2])==0){
                imagesC[2].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[2][x2]);
                //x2++;
                imagesNames[2].push_back(currentName);
            }else if(pngFiles[i].find(camNames[3])==0){
                imagesC[3].push_back(imageNow);
                imageNow.copyTo(imagesC[3][x3]);
                //x3++;
                imagesNames[3].push_back(currentName);
            }else if(pngFiles[i].find(camNames[4])==0){
                imagesC[4].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[4][x4]);
                //x4++;
                imagesNames[4].push_back(currentName);
            }else if(pngFiles[i].find(camNames[5])==0){
                imagesC[5].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[5][x5]);
                //x5++;
                imagesNames[5].push_back(currentName);
            }else if(pngFiles[i].find(camNames[6])==0){
                imagesC[6].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[6][x6]);
                //x6++;
                imagesNames[6].push_back(currentName);
            }else if(pngFiles[i].find(camNames[7])==0){
                imagesC[7].push_back(Mat::zeros(h, w, CV_32FC3));
                imageNow.copyTo(imagesC[7][x7]);
                //x7++;
                imagesNames[7].push_back(currentName);
            }else{
                cout << "Error when separating the images by cameras occured." << endl;
            }
        }
        cout << "Images storing completed! \n";
    }
 
    h = imagesC[0][0].rows;
    w = imagesC[0][0].cols;
    
    // READ SILHOUETTES PBM
    
    glob(silhouettesDir.operator+=("/*.pbm"),pbmFiles,false); //Get all the pbm files in the input silhouettes directory
    Mat silNow;
    string currSilN;
    if (pbmFiles.size() == 0){
        cerr << "\nERROR: No pbm images were found.";
        return EXIT_FAILURE;
    }else{
        int x0 = 0,x1 = 0,x2 = 0,x3 = 0,x4 = 0,x5 = 0,x6 = 0,x7 = 0;
        for(int i = 0; i < pbmFiles.size(); i++){
            silNow = imread(pbmFiles[i],IMREAD_UNCHANGED);
            silNow.convertTo(silNow, CV_32FC1);
            currSilN = pbmFiles[i]; // name of the actual silhouette
            if(pbmFiles[i].find(camSilNames[0])==0){
                silC[0].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[0][x0]);
                //x0++;
                silNames[0].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[1])==0){
                silC[1].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[1][x1]);
                //x1++;
                silNames[1].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[2])==0){
                silC[2].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[2][x2]);
                //x2++;
                silNames[2].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[3])==0){
                silC[3].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[3][x3]);
                //x3++;
                silNames[3].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[4])==0){
                silC[4].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[4][x4]);
                //x4++;
                silNames[4].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[5])==0){
                silC[5].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[5][x5]);
                //x5++;
                silNames[5].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[6])==0){
                silC[6].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[6][x6]);
                //x6++;
                silNames[6].push_back(currSilN);
            }else if(pbmFiles[i].find(camSilNames[7])==0){
                silC[7].push_back(Mat::zeros(h, w, CV_32FC1));
                silNow.copyTo(silC[7][x7]);
                //x7++;
                silNames[7].push_back(currSilN);
            }else{
                cout << "Error when separating the silhouettes by cameras occured." << endl;
            }
        }
        cout << "Silhouettes storing completed! \n";
        
    }
    
    // les siluetes de cada camera son diferents -> okey.
    
    // LOAD CAMERA FILES and MATRICES P
    // Read xml files to store matrices P from the cameras
    glob(calibrationDir.operator+=("/*.xml"),xmlFiles,false); //Get all the XML files in the calibration directory
    // - Compute camera parameters associated to each image (and camera) - projeccions.

    int f = 0;
    Mat S,U,VT;
    Mat center;
    vector<Mat> matricesP; // Known Camera Projections
    vector<Mat> centers(8, Mat(4,1,CV_32F));  //vectors of camera centers
    for(int i=0; i < xmlFiles.size();i++){
        vector<float> valuesP;
        XMLDocument doc;
        doc.LoadFile(xmlFiles[i].c_str());
        auto sht = doc.FirstChildElement("camera");
        const char *str;
        str = sht->GetText(); // All floats together
        // get float values
        istringstream ss(str);
        copy(
             istream_iterator<float> (ss ),
             istream_iterator<float> (),
             back_inserter(valuesP)
             );
        Mat m(3,4,CV_32F);
        f=0;
        float lastvalue = valuesP.back();
        for(int r = 0; r < m.rows; r++){
            for(int c = 0; c < m.cols; c++){
                m.at<float>(r,c) = valuesP[f]/lastvalue;
                f++;
            }
        }
        matricesP.push_back(m);
        SVD::compute(m,S,U,VT,SVD::FULL_UV);
        Mat V = VT.t(); //the transpose
        // C is the last column of V
        center = V(Range(0,3),Range(3,4))/V.at<float>(3,3);
        center.copyTo(centers[i]);
    }
    S.deallocate();
	U.deallocate();
	VT.deallocate();
	center.deallocate();
    
	cout << "Centers and Matrices P of all cameras stored. \n";
	
    int ncam = matricesP.size();
    
    int64 e2 = getTickCount();
    double timedataset = (e2 - e1)/getTickFrequency();
    cout << "Dataset read in " << floor(timedataset/60) << " min " << fmod(timedataset,60) << " s\n" << endl;
    
	/* --------------------------------- DEPTH MAP ESTIMATION ---------------------------------
	 
     - Compute the Confidence Volume (V).
	 - Compute descriptors (Di,Dj) with DAISY.
	 - Photoconsistency measure (p).
	 - Depth prediction (d).  */
	
	
	// *************** LOAD VBOX *********************
	
	// 1. Define 3D volume of voxels (big enough)
	// 2. Project each voxel to the images and silhouettes to see if they accomplish the conditions of V
	// 3. If voxel inside de volume --> 1. Set voxel to 0 on the contrary.
	
    int Nx = 130; // 70 per vs 0.025, 80 per vs = 0.02, 110 per 0.018, 120 per mes petit
    int Ny = 90; // 70 per vs 0.025, 80 per vs = 0.02, 100 per 0.018, 110 per mes petit
    int Nz = 185; // 80 per vs = 0.025 , 95 per vs = 0.02, 130 per 0.018, 150 per mes petit
    float voxelsize = .15;
                            // 0.02 triga uns 1:30 h
                            // 0.01 triga 11h i pico, Nx = 130, Ny = 90, Nz = 185
                            // 0.015 triga 3h i mitja
    float Ox = -0.3;
    float Oy = -1.8; // -2.3
    float Oz = -0.15;
	
//	PARAMETERS FOR THE VOLUME OF VOXELS
	
    int alpha = matricesP.size(); // Belonging to all cameras -x
    int beta = matricesP.size()-1;    // Belonging to all silhouettes - x
	VolumeBox vbox(Ox,Oy,Oz,Nx,Ny,Nz, voxelsize);
	
	// Compute Voxel Volume
	
	int nx = vbox.nx;
	int ny = vbox.ny;
	int nz = vbox.nz;
	int vol[] = {nx,ny,nz};
	Mat voxels_projected_volume(3,vol,CV_32F);

	int imgId = 0; // When we want to compute this for every time instant k, a loop will be performed for every imgId
	int x,y,z;
    Mat sil1 = silC[0][0];
	Mat currentSil(sil1.rows,sil1.cols,CV_32FC1);
	
	int N = nx*ny*nz;
	vector<float> idxs_goodvoxels;
    idxs_goodvoxels.reserve(N);
	Mat voxel_4D(4,1,CV_32F);
    omp_set_num_threads(8);

    for (int z=0; z < nz; z++) {
		for (int y=0; y < ny; y++) {
			for (int x=0; x < nx; x++) {
				
				// Current voxel coordinates in the 3D space (world coord.)
                float xcoord = x*voxelsize + Ox + voxelsize/2;
                float ycoord = y*voxelsize + Oy + voxelsize/2;
                float zcoord = z*voxelsize + Oz + voxelsize/2;
                Mat voxel_4D = 1.0*(Mat_<float>(4,1,CV_32F) << xcoord,ycoord,zcoord,1.0);

				int image_counter = 0;
				int sil_counter = 0;
				for (int camId=0; camId < matricesP.size(); camId++) {
					imgId = 0;
					currentSil = silC[camId][imgId];
					int w = imagesC[camId][imgId].cols;
					int h = imagesC[camId][imgId].rows;
					// Project the voxel from the 3D space to the images
					Mat P = matricesP[camId];
					Mat projection = P*voxel_4D;
					//We get the point in homog coord. (image coord.)
					float xp = projection.at<float>(0);
					float yp = projection.at<float>(1);
					float zp = projection.at<float>(2);
					// Get the cartesian coord
					int xp2d = cvRound(xp/zp);
					int yp2d = cvRound(yp/zp);
					// If voxel inside of imagesC[camId][imgId] --> image_counter++;
					// If voxel inside of silC[camId][imgId] --> sil_counter++;
					if(xp2d >= 0 && xp2d < w && yp2d >= 0 && yp2d < h){
                        image_counter++;
						int value = currentSil.at<float>(yp2d, xp2d);
						if(value == 255){
							sil_counter++;
						}
					}
				}
                if(sil_counter >= beta && image_counter >= alpha) {
                    voxels_projected_volume.at<int>(x,y,z) = 1; // set voxel to 1
                    // Current voxel index (from 3D array to 1D array)
                    float idx = (float) x + nx * (y + ny * z);
					idxs_goodvoxels.push_back(idx);
                }else {
                    voxels_projected_volume.at<int>(x,y,z) = 0; // set voxel to 0
                }
			}
		}
	}
    vbox.toVTK("/Users/helena/Documents/TFG/volume.vtk", voxels_projected_volume);
	cout << "\n";
    cout << " Voxel size: " << voxelsize <<", numero de voxels del volum = " << idxs_goodvoxels.size() << endl;
	cout << "Voxel Volume computed! \n";
    
    int64 e3 = getTickCount();
    double timevisualh = (e3 - e2)/getTickFrequency();
    cout << "Visual Hull calculated in " << floor(timevisualh/60) << " min " << fmod(timevisualh,60) << " s\n" << endl;

    
    // If we want to blur the images before computing DAISY descriptors, we use a Gaussian filter:
    /*vector<Mat> blurredImages(8,Mat());
    string blurred_imagesnames[matricesP.size()];
    for (int i = 0; i < matricesP.size(); i++) {
        Mat blurredimage;
        GaussianBlur(imagesC[i][imgId], blurredimage,Size(5,5), 1,0);
        blurredImages[i] = blurredimage;
        std::ostringstream name;
        name << "../../primeres_imatges_de_cada_camera/blurred_0" << i << ".png";
        string s = name.str();
        imwrite(s, blurredimage);
        blurred_imagesnames[i] = s;
    }*/
    
  
	// Compute Descriptors of the images.
	
	int verbose_level=0;
	// default values:
	int rad   = 15; // Distance from the center pixel to the outer most grid point.
	int radq  =  3; // Number of convolved orientation layers with different standard deviations
	int thq   =  8; //  Number of histograms at a single layer
	int histq =  8; // Number of bins in the histogram
	
	daisy* desc = new daisy();

    vector< vector< vector< array<float,200> > > > imageDescriptors(8, vector< vector< array<float,200> > >(h, vector< array<float,200> >(w)));
    Mat desc_images[ncam];
	int ch = 3;
    float desc_size[ncam];

    vector<Mat> voxel_luts_xp2d(ncam,Mat(3, vol, CV_32FC1));
    vector<Mat> voxel_luts_yp2d(ncam,Mat(3, vol, CV_32FC1));
    
    omp_set_num_threads(8);
#pragma private(camId,desc)
	for(int camId = 0; camId < matricesP.size(); camId++){
        
        int imh, imw;
		uchar* im = NULL;
        string filename = imagesNames[camId][imgId];
        //string filename = blurred_imagesnames[camId];
		load_image(filename,im,imh,imw,ch);
		desc->set_image(im, imh, imw);
		desc->verbose(verbose_level);
		desc->set_parameters(rad, radq, thq, histq);
		desc->initialize_single_descriptor_mode();
		desc->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED)
		desc->normalize_descriptors();
        desc_size[camId] = desc->descriptor_size();

        Mat_<int> auxiliarMat_xp2d = Mat::zeros(3, vol, CV_32FC1);
        Mat_<int> auxiliarMat_yp2d = Mat::zeros(3, vol, CV_32FC1);
		
    #pragma omp parallel for collapse(3)
		for (int z = 0; z < nz; z++) {
			for (int y = 0; y < ny; y++) {
				for (int x = 0; x < nx; x++) {
                    float idx =(float) x + nx * (y + ny * z);
                    // check for voxels inside the visual hull (VH)
                    if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()){
                        // Current voxel coordinates in the 3D space
                        float xcoord = x*voxelsize + Ox + voxelsize/2;
                        float ycoord = y*voxelsize + Oy + voxelsize/2;
                        float zcoord = z*voxelsize + Oz + voxelsize/2;
                        // Project the voxel from the 3D space to the images
                        Mat P = matricesP[camId];
                        Mat projection = P*(Mat_<float>(4,1) << xcoord,ycoord,zcoord,1.0);
                        //We get the point in homog coord.
                        float xp = projection.at<float>(0);
                        float yp = projection.at<float>(1);
                        float zp = projection.at<float>(2);
                        // Get the cartesian coord
                        int xp2d = cvRound(xp/zp);
                        int yp2d = cvRound(yp/zp);
                        
                        if(xp2d >= 0 && xp2d < imw && yp2d >= 0 && yp2d < imh){
                            float* thor = new float[desc->descriptor_size()]; 
                            desc->get_descriptor(yp2d,xp2d,thor);
                            auxiliarMat_xp2d.at<int>(x,y,z) = xp2d;
                            auxiliarMat_yp2d.at<int>(x,y,z) = yp2d;
                            
							if(thor){ // ara ja no sé si cal el if
                                for(int id=0; id<desc->descriptor_size(); id++) {
                                    imageDescriptors[camId][yp2d][xp2d][id] = thor[id];
                                }
                            }
						}
                    }
				} // end for x
			} // end for y
		} // end for z
        
        voxel_luts_xp2d[camId] = auxiliarMat_xp2d;
        voxel_luts_yp2d[camId] = auxiliarMat_yp2d;
        auxiliarMat_yp2d.release();
        auxiliarMat_xp2d.release();
 
        desc->reset();
	}

	cout << "voxelDescriptors calculats.\n";
    delete desc;
    int64 e4 = getTickCount();
    double timedaisy = (e4 - e3)/getTickFrequency();
    cout << "Daisy descriptors and voxel coordinates in " << floor(timedaisy/60) << " min " << fmod(timedaisy,60) << " s\n" << endl;
    
	int i = 0;
	int j = 0;
    
    int arraysize =nx*ny*nz;
    
    vector< vector<cv::Mat> > g(ncam,vector<Mat>(ncam,(-1.0)*Mat::zeros(3,vol,CV_32F)));
    vector< vector< vector<float> > > gx(arraysize, vector< vector<float> >(ncam,vector<float>(ncam)));
    
	float wjnorm;
    vector< vector< vector<float> > > wj(arraysize, vector< vector<float> >(ncam,vector<float>(ncam)));
    vector< vector<float> > wk(arraysize, vector<float>(ncam));
    
    vector< vector<float> > pidx(arraysize,vector<float>(ncam));
    
#pragma omp parallel for collapse(3) private(x,y,z)
	for (int z = 0; z < nz; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				float idx =(float) x + nx * (y + ny * z);
				if (find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()) {
					for (int i = 0; i < matricesP.size(); i++) {
                        for (int j = 0; j < matricesP.size(); j++) {
							if (i!=j) {
                                float xcoord = x*voxelsize + Ox + voxelsize/2;
                                float ycoord = y*voxelsize + Oy + voxelsize/2;
                                float zcoord = z*voxelsize + Oz + voxelsize/2;
                                
                                vector<float> xci(3);
								xci[0] = centers[i].at<float>(0)-xcoord;
								xci[1] = centers[i].at<float>(1)-ycoord;
								xci[2] = centers[i].at<float>(2)-zcoord;
								vector<float> xcj(3);
								xcj[0] = centers[j].at<float>(0)-xcoord;
								xcj[1] = centers[j].at<float>(1)-ycoord;
								xcj[2] = centers[j].at<float>(2)-zcoord;
								
								//computing the angle:
								float dot = xci[0]*xcj[0] + xci[1]*xcj[1] + xci[2]*xcj[2];
								float lSi = xci[0]*xci[0] + xci[1]*xci[1] + xci[2]*xci[2];
								float lSj = xcj[0]*xcj[0] + xcj[1]*xcj[1] + xcj[2]*xcj[2];
								float cosangle = dot/sqrt(lSi+lSj);
                                wj[idx][i][j] = cosangle;
                                
                                if(cosangle > 0.7){
                                    wk[idx][i] += cosangle;
                                    int xp2d_i = voxel_luts_xp2d[i].at<int>(x,y,z);
                                    int yp2d_i = voxel_luts_yp2d[i].at<int>(x,y,z);
                                    int xp2d_j = voxel_luts_xp2d[j].at<int>(x,y,z);
                                    int yp2d_j = voxel_luts_yp2d[j].at<int>(x,y,z);
                                    
                                    array<float, 200> descriptor_i = imageDescriptors[i][yp2d_i][xp2d_i];
                                    array<float, 200> descriptor_j = imageDescriptors[j][yp2d_j][xp2d_j];
                                  
                                    double dist = norm(descriptor_i, descriptor_j, NORM_L2); //EUCLIDEAN DISTANCE
                                    gx[idx][i][j] = dist;
                                    //g[i][j].at<float>(x,y,z) = dist;
								}
                            } // end if(i!=j)
						} // end j
                    
					} //end i
                    
				} // end if V[x] == 1
				
			} // x
		} // y
	} // z

	//vbox.toVTK_float("/Users/helena/Documents/TFG/g_x.vtk", g[4][5]);
	cout << "pairwaise photometric discrepancy g calculada.\n";
    int64 e5 = getTickCount();
    double timeg = (e5 - e4)/getTickFrequency();
    cout << "gij calculated in " << floor(timeg/60) << " min " << fmod(timeg,60) << " s\n" << endl;
    
    
    float sumpi;
    float sigma = 0.7;
    
#pragma omp parallel for collapse(3) private(x,y,z) // en teoria si que va be aixi
	for (int z = 0; z < nz; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				float idx = x + nx * (y + ny * z);
				if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()) {
					for(int i = 0; i < ncam; i++){
						sumpi = 0;
						for(int j = 0; j<ncam; j++){
							if(i!=j){
								//compute all the contributions from j cameras to the voxel x
								if (wj[idx][i][j] > 0.7) {
                                    wjnorm = wj[idx][i][j]/wk[idx][i];
									sumpi += wjnorm*exp((-pow(gx[idx][i][j],2))/pow(2*sigma,2)); // wnorm*e^(-g[i][j][x]²/(2sigma^2))
								}else{
									sumpi += 0.0;
								}
							} // end if(i!=j)
						} //end j
                        pidx[idx][i] = sumpi;
                    } // end i
				} // end if V[x] == 1
			} // end x
		} // end y
	} // end z
    
	cout << "Photoconsistency measure calculada.\n";
    //vbox.toVTK_float("/Users/helena/Documents/TFG/p_i.vtk", p[3]);
    
    int64 e6 = getTickCount();
    double timephoto = (e6 - e5)/getTickFrequency();
    cout << "pi calculated in " << floor(timephoto/60) << " min " << fmod(timephoto,60) << " s\n" << endl;

	
	// COMPUTE DEPTH PREDICTION di(p)
		
	vector< vector< vector< vector<Mat> > > > lutsRay(ncam,vector< vector< vector<Mat> > >(h,vector< vector<Mat> >(w,vector<Mat>())));
	int s[3] = {nx,ny,nz};
	vector<Mat> distances(centers.size(),Mat_<int>(3, s, CV_32F)); // guarda per cada voxel la seva distància al centre de cada càmera

	// les distancies dels voxels respecte les càmeres i els rays look up table es poden calcular fora.
	distances = vbox.create_distance_lut(centers, idxs_goodvoxels);
    vbox.toVTK_float("/Users/helena/Documents/TFG/distanceLut.vtk", distances[2]);
    cout << "Distances of voxels to each one of the cameras computed. \n";
    
    int64 e7 = getTickCount();
    double timedistances = (e7 - e6)/getTickFrequency();
    cout << "Distances LUT calculated in " << floor(timedistances/60) << " min " << fmod(timedistances,60) << " s\n" << endl;
    
    // 1. create ray lut for each image and pixel p
    int indexCam;
#pragma omp parallel for private(indexCam)
    for (int indexCam=0; indexCam<matricesP.size(); indexCam++){ // for each image
        #pragma omp parallel for collapse(3)
        for(int z=0; z<nz; z++){
            for(int y=0; y<ny; y++){
                for(int x=0; x<nx; x++){
                    float idx = (float) x + nx * (y + ny * z);
                    if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()){ // if it is true, then:
//
                        int xp2d = voxel_luts_xp2d[indexCam].at<int>(x,y,z);
                        int yp2d = voxel_luts_yp2d[indexCam].at<int>(x,y,z);
                        
                        Mat tmp(2,1,CV_32FC1);
                        tmp.at<float>(0) = idx;
                        tmp.at<float>(1) = distances[indexCam].at<float>(x,y,z); //distanceLut vector --> get the distance of the current voxel
                        lutsRay[indexCam][yp2d][xp2d].push_back(tmp);
                    }
                }
            }
        }
    }
    
    // 2. sort each pixel ray according to the distance from voxels to the camera
#pragma omp parallel for private(indexCam)
    for (int indexCam=0; indexCam<matricesP.size(); indexCam++){
        for(int r=0; r<h; r++){
            for(int c=0; c<w; c++){
                unsigned long n_voxels = lutsRay[indexCam][r][c].size();
                if(n_voxels != 0){
                    sort(lutsRay[indexCam][r][c].begin(),lutsRay[indexCam][r][c].end(), vbox.compare);
                }
            }
        }
    }
    cout << "Rays computed." << endl;

    int64 e8 = getTickCount();
    double timerays = (e8 - e7)/getTickFrequency();
    cout << "Rays LUT calculated in " << floor(timerays/60) << " min " << fmod(timerays,60) << " s\n" << endl;
    
    vector<Mat> img_depth(8,Mat::zeros(h, w, CV_32FC1));
    float pmax = 20;
    float thr_photo = 0.68;
    
    // Compute depth map for every view
    int row,col;
#pragma omp parallel for private(indexCam)
    for(int indexCam = 0; indexCam < matricesP.size(); indexCam++){
        Mat img_depth_auxiliar = Mat::zeros(h, w, CV_32FC1);
        int tau_under_counter = 0;
        int tau_greater_counter = 0;
        #pragma omp parallel private(row,col)
        for (int row = 0; row < h; row+=1) {
			for (int col = 0; col < w; col+=1) {
				unsigned long n_voxels = lutsRay[indexCam][row][col].size();
                if(n_voxels != 0){
//					2. Compute dv(pixel) - distance dv of the first depth value in the visual ray
					float dv = 0;
					int voxel_dv = 0;
					int v = 0; // current voxel
					while (dv == 0) {
                        dv = lutsRay[indexCam][row][col][v].at<float>(1);
                        voxel_dv = v;
					}
//					3. Compute dmax - sum of values of p[sil] along the visual ray until pmax
                    float sump = 0.0;
					int voxel = 0;
					int nvoxel_max = 0; // voxel where we find dmax
					while (sump < pmax) {
						if(voxel<n_voxels){
							float idx = lutsRay[indexCam][row][col][voxel].at<float>(0);
							float photoconsistency = pidx[idx][indexCam];
                            sump += photoconsistency;
							nvoxel_max = voxel;
                            #pragma omp atomic
							voxel++;
						}else{
							sump = pmax;
						}
					}
//					4. Loof for the maximum of p_sil[ray] between dv(p) and dmax, store depth of the voxel where the maximum is found.
					float p_imgmax = 0; // maximum of photo-consistency
					float d_imgmax = 0; // distance where the maximum is found
					for (int vox = voxel_dv; vox < nvoxel_max+1; vox++) {
						float idx = lutsRay[indexCam][row][col][vox].at<float>(0);
						float dist = lutsRay[indexCam][row][col][vox].at<float>(1);
                        float p_img = pidx[idx][indexCam];
						if (p_img > p_imgmax) {
							p_imgmax = p_img;
							d_imgmax = dist;
						}
					}
//					5. Check if the maximum p_silmax is under or grater than thr_photo:
//						5.1. If p_silmax < thr_photo --> sil_depth[pixel] = dv(p)
//						5.2. If p_silmax > thr_photo --> sil_depth[pixel] = d_imgmax.
					if(p_imgmax < thr_photo){
						img_depth_auxiliar.at<float>(row,col) = dv;
					}else{
                        img_depth_auxiliar.at<float>(row,col) = d_imgmax;
                    }
                } else img_depth_auxiliar.at<float>(row,col) = NAN; // If the pixel has no depth value
			} // end cols
		} // end rows
        img_depth[indexCam] = img_depth_auxiliar;
        img_depth_auxiliar.release();
	} // end cams
    cout << "Depth images without filtering computed" << endl;
   
    // Store raw depth map images
    
    double min0, max0, min1, max1,min2, max2, min3, max3, min4, max4, min5, max5, min6, max6, min7, max7;
    minMaxLoc(img_depth[0], &min0, &max0,NULL,NULL,img_depth[0]>0);
    cout << "min0: "<< min0 << ", max0 " << max0 << endl;
    Mat imgdepth_00(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_00.at<float>(row, col) = ((img_depth[0].at<float>(row,col) - min0)/(max0-min0))*255;
        }
    }
    minMaxLoc(img_depth[1], &min1, &max1,NULL,NULL,img_depth[1]>0 );
    cout << "min1: "<< min1 << ", max1 " << max1 << endl;
    Mat imgdepth_01(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_01.at<float>(row, col) = ((img_depth[1].at<float>(row,col) - min1)/(max1-min1))*255;
        }
    }
    minMaxLoc(img_depth[2], &min2, &max2,NULL,NULL,img_depth[2]>0 );
    cout << "min2: "<< min2 << ", max2 " << max2 << endl;
    Mat imgdepth_02(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_02.at<float>(row, col) = ((img_depth[2].at<float>(row,col) - min2)/(max2-min2))*255;
        }
    }
    minMaxLoc(img_depth[3], &min3, &max3,NULL,NULL,img_depth[3]>0 );
    cout << "min3: "<< min3 << ", max3 " << max3 << endl;
    Mat imgdepth_03(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_03.at<float>(row, col) = ((img_depth[3].at<float>(row,col) - min3)/(max3-min3))*255;
        }
    }
    minMaxLoc(img_depth[4], &min4, &max4,NULL,NULL,img_depth[4]>0 );
    cout << "min4: "<< min4 << ", max4 " << max4 << endl;
    Mat imgdepth_04(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_04.at<float>(row, col) = ((img_depth[4].at<float>(row,col) - min4)/(max4-min4))*255;
        }
    }
    minMaxLoc(img_depth[5], &min5, &max5,NULL,NULL,img_depth[5]>0 );
    cout << "min5: "<< min0 << ", max5 " << max5 << endl;
    Mat imgdepth_05(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_05.at<float>(row, col) = ((img_depth[5].at<float>(row,col) - min5)/(max5-min5))*255;
        }
    }
    minMaxLoc(img_depth[6], &min6, &max6,NULL,NULL,img_depth[6]>0 );
    cout << "min6: "<< min6 << ", max6 " << max6 << endl;
    Mat imgdepth_06(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_06.at<float>(row, col) = ((img_depth[6].at<float>(row,col) - min6)/(max6-min6))*255;
        }
    }
    minMaxLoc(img_depth[7], &min7, &max7,NULL,NULL,img_depth[7]>0 );
    cout << "min7: "<< min7 << ", max7 " << max7 << endl;
    Mat imgdepth_07(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_07.at<float>(row, col) = ((img_depth[7].at<float>(row,col) - min7)/(max7-min7))*255;
        }
    }
        //In order to see the images we have to transform the depth to 0-255, only to visualize
    
//    Mat imgdepth_00;
//    normalize(img_depth[0], imgdepth_00, 0, 255, NORM_MINMAX, CV_8UC(1));
//    minMaxLoc(img_depth[0], &min0, &max0);
//    img_depth[0].convertTo(imgdepth_00, CV_32F,255.0/(max0-min0), -255.0/min0);


    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_00.png", imgdepth_00 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_01.png", imgdepth_01 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_02.png", imgdepth_02 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_03.png", imgdepth_03 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_04.png", imgdepth_04 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_05.png", imgdepth_05 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_06.png", imgdepth_06 );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_07.png", imgdepth_07 );

    
        // APLICAR BILATERAL FILTER PER CADA IMATGE
    vector<Mat> img_depth_filtered(8,Mat::zeros(h, w, CV_32FC1)); // la depth son floats? ints?
    float sigma_s = 6; //domain parameter for spatial kernel
    int window_size = 5; // en realitat sera una finestra de windowsize*2xwindowsize*2
    float sigma_r = 30; //range parmeter for intensity kernel
    float sigma_bf = 20;
    int max_it = 6;
    int extra_it = 4;
    int camId;
#pragma omp parallel for private(camId)
    for (int camId = 0; camId < matricesP.size(); camId++) {
        Mat current_depthImage_original = img_depth[camId].clone();  // image to get the original depth values
        Mat current_depthImage = img_depth[camId].clone(); // image to modify
        Mat current_sil = silC[camId][imgId];
        Mat current_rgbImage = imagesC[camId][imgId];
        //Mat current_rgbImage = blurredImages[camId];
        Mat imageFiltered = Mat::zeros(h,w,CV_32FC1);
        int it = 0;
        while (it<max_it+extra_it) {
            #pragma omp parallel private(row,col)
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col ++) {
                    auto n_voxels = lutsRay[camId][row][col].size();
                    int value = current_sil.at<float>(row, col);
                    bool condition = false;
                    if (it <= max_it-2) {
                        if (value == 255 && n_voxels!= 0) {
                            condition = true;
                        }
                    }else{
                        if (value == 255 || n_voxels != 0) {
                            condition = true;
                        }
                    }
                    if(condition){
                        float current_depth_value = current_depthImage_original.at<float>(row, col);
                        if (it > max_it) {
                            current_depth_value = 0;
                            sigma_r = sigma_bf;
                        }
                        if (current_depth_value == 0 || isnan(current_depth_value)) {
                            if (it!=0){
                                current_depthImage = imageFiltered.clone();
                            }
                            // Adjusting the window size
                            int imin=max(row-window_size,0);
                            int imax=min(row+window_size,h);
                            int jmin=max(col-window_size,0);
                            int jmax=min(col+window_size,w);
                           
                            float sum_weights = 0;
                            float sum_depths = 0;
                            float spatial_value = 0;
                            float intensity_value = 0;
                            float current_weight = 0;
                            float current_pixel_depth = 0;
                            float pixel_depth_weighted = 0;
                            
                            for (int k = imin; k < imax; k++) { // row
                                for (int l = jmin; l < jmax; l++) { //  column
                                    if (lutsRay[camId][k][l].size() != 0) { // pixels with no voxel projection do not contribute
                                        spatial_value = (pow(row - k,2) + pow(col - l, 2))/(2*pow(sigma_s, 2));
                                        Point3_<float>* c_ij = current_rgbImage.ptr<Point3_<float> >(row,col); // original RGB image
                                        Point3_<float>* c_kl = current_rgbImage.ptr<Point3_<float> >(k,l);
                                        double R = c_ij->x - c_kl->x;
                                        double G = c_ij->y - c_kl->y;
                                        double B = c_ij->z - c_kl->z;
                                        //((R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2)
                                        float norma = pow(R, 2) + pow(G, 2) + pow(B, 2);
                                        intensity_value = norma/(2*pow(sigma_r, 2));
                                        current_weight = exp( - spatial_value - intensity_value);
                                        current_pixel_depth = current_depthImage.at<float>(k,l);
                                        pixel_depth_weighted = current_pixel_depth*current_weight;
                                        sum_weights += current_weight;
                                        sum_depths += pixel_depth_weighted;
                                        norma = 0;
                                    }
                                }
                            }
                            // Now we can compute the depth value at pixel (row,col) with respect to their (k,l)
                            if (sum_weights !=0) {
                                imageFiltered.at<float>(row, col) = sum_depths/sum_weights;
                            }else{
                                imageFiltered.at<float>(row, col) = NAN; // to define pixels with no depth
                            }
                        }else{
                            imageFiltered.at<float>(row, col) = current_depthImage_original.at<float>(row, col);
                        }
                    }else{
                        imageFiltered.at<float>(row, col) = NAN;
                    }
                } // end col
            } // end row
            #pragma omp atomic
            it++;
        }
        img_depth_filtered[camId] = imageFiltered;
        imageFiltered.release();
        current_depthImage.release();
        current_rgbImage.release();
        
    } // end camId
    
     // SAVE FILTERED IMAGES
    
    Mat imgdepth_00f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_00f.at<float>(row, col) = ((img_depth_filtered[0].at<float>(row,col) - min0)/(max0-min0))*255;
        }
    }
    Mat imgdepth_01f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_01f.at<float>(row, col) = ((img_depth_filtered[1].at<float>(row,col) - min1)/(max1-min1))*255;
        }
    }
    Mat imgdepth_02f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            imgdepth_02f.at<float>(row, col) = ((img_depth_filtered[2].at<float>(row,col) - min2)/(max2-min2))*255;
        }
    }
    Mat imgdepth_03f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col= 0; col < w; col++) {
            imgdepth_03f.at<float>(row, col) = ((img_depth_filtered[3].at<float>(row,col) - min3)/(max3-min3))*255;
        }
    }
    Mat imgdepth_04f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col=0; col < w; col++) {
            imgdepth_04f.at<float>(row, col) = ((img_depth_filtered[4].at<float>(row,col) - min4)/(max4-min4))*255;
        }
    }
    Mat imgdepth_05f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col=0; col < w; col++) {
            imgdepth_05f.at<float>(row, col) = ((img_depth_filtered[5].at<float>(row,col) - min5)/(max5-min5))*255;
        }
    }
    Mat imgdepth_06f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col=0; col < w; col++) {
            imgdepth_06f.at<float>(row, col) = ((img_depth_filtered[6].at<float>(row,col) - min6)/(max6-min6))*255;
        }
    }
    Mat imgdepth_07f(h,w,CV_32F);
    for (int row = 0; row < h; row++) {
        for (int col=0; col < w; col++) {
            imgdepth_07f.at<float>(row, col) = ((img_depth_filtered[7].at<float>(row,col) - min7)/(max7-min7))*255;
        }
    }

    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_00f.png", imgdepth_00f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_01f.png", imgdepth_01f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_02f.png", imgdepth_02f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_03f.png", imgdepth_03f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_04f.png", imgdepth_04f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_05f.png", imgdepth_05f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_06f.png", imgdepth_06f );
    imwrite( "/Users/helena/Documents/TFG/Depth_images/depth_image_07f.png", imgdepth_07f );
    
    cout << "Depth images filtered. \n";
    
    int64 e9 = getTickCount();
    double timemaps = (e9 - e8)/getTickFrequency();
    cout << "Depth maps calculated in " << floor(timemaps/60) << " min " << fmod(timemaps,60) << " s\n" << endl;
    
	/* --------------------------------- SHAPE ESTIMATION ---------------------------------
	 - Compute Spatial Integration (TSDF) */
    
    // consider a single frame and the spatial integration of the depth maps di for all cameras at that frame
    
    Mat TD(3,vol,CV_32F); // weighted average of all camera predictions Fi(x)
    float mu = 1.1;
    float empty = -mu;
    vector<float> TD_array; // to compute the minimum and maximum values
    TD_array.reserve(N);
    
    fstream file;
    file.open("/Users/helena/Documents/TFG/TD_information3.txt");
    file.close();

    FILE *tdfile = NULL;
    tdfile = fopen("/Users/helena/Documents/TFG/TD_information3.txt","w");
    fprintf(tdfile,"Information about how TD is formed. \n");
    int counter_nan = 0;
    int counter_zero = 0;
    
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float idx = (float) x + nx * (y + ny * z);
                if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()){
                    int camera_counter = 0;
                    int empty_counter = 0;
                    float p_norm = 0;
                    float sum_F = 0;
                    for (int i = 0; i < matricesP.size(); i++) { // for each camera
                        int col = voxel_luts_xp2d[i].at<int>(x,y,z);
                        int row = voxel_luts_yp2d[i].at<int>(x,y,z);
                        float di = img_depth_filtered[i].at<float>(row,col); // depth estimated at pixel (row,col) for the camera i
                        if(isnan(di)){
                            counter_nan++;
                        }else{
                            if (di != 0) { // check that the depth is defined at the pixel
                                float eta = di - distances[i].at<float>(x,y,z);
                                if(eta >= -mu){
                                    float value = min(mu, eta);
                                    float current_p = pidx[idx][i]; // current photoconsistency of the voxel
                                    sum_F += current_p*value;
                                    p_norm += current_p;
                                    // An alternative to the average mean:
                                    //sum_F += value;
                                    //p_norm += 1; //
                                }else{
                                    empty_counter++;
                                }
                                camera_counter++;
                            }else{
                                counter_zero++;
                            }
                        }
                    } // camera i
                    
                    // fill the TD(x)
                    if (camera_counter == 0) { // si cap camera contribueix a x pero x dins de V,
                        TD.at<float>(x, y, z) = -mu/2; // TD(x) < 0 --> x es troba dins del volum perque hem comprovay el seu index
                        TD_array.push_back(-mu/2);
                        //#pragma omp critical
                        fprintf(tdfile,"camera_counter == 0,             x: %d y: %d z: %d , p_norm= %f, TD_value = %f\n",x,y,z,p_norm,-mu/2);
                    }else if(empty_counter == matricesP.size()){ // realment mai entra aquí
                        TD.at<float>(x, y, z) = empty; //(-1)
                        TD_array.push_back(empty);
                        //#pragma omp critical
                        fprintf(tdfile,"empty_counter == matrices.P(),   x: %d y: %d z: %d , p_norm= %f, TD_value = %f\n",x,y,z,p_norm,-mu);
                    }else if (p_norm == 0){// vol dir que alguna camera si que contribueix pero que no tenen info de fotoconsistència... llavors posar-ho a -mu o 0???
                        TD.at<float>(x, y, z) = empty; //provem d'assignar-li -mu
                        TD_array.push_back(empty);
                        //#pragma omp critical
                        fprintf(tdfile,"p_norm == 0,                     x: %d y: %d z: %d , p_norm= %f, TD_value = %f\n",x,y,z,p_norm,-mu);
                    }else{
                        TD.at<float>(x, y, z) = sum_F/p_norm; // com que sum_F < 0, indica que x esta dins del volum
                        TD_array.push_back(sum_F/p_norm);
                        //#pragma omp critical
                        fprintf(tdfile,"sum_F/p_norm,                    x: %d y: %d z: %d , p_norm= %f, TD_value = %f\n",x,y,z,p_norm,sum_F/p_norm);
                    }
                    // si el voxel no està dins del volum V, no té TD, no?
                }else { // està fora del volum si o si, ha de ser valor positiu
                    TD.at<float>(x, y, z) = mu; //(1)
                    TD_array.push_back(mu);
                }
    
            }
        }
    }
    
    fclose(tdfile);
    cout << "counter_nan: " << counter_nan << endl;
    cout << "counter_zero: " << counter_zero << endl;
    
    vbox.toVTK_float("/Users/helena/Documents/TFG/TD.vtk", TD);
    vbox.writeIsoSurfaceVTK("/Users/helena/Documents/TFG/TD_newrange.vtk", TD, TD_array);
    cout << "Spatial Integration TD computed! \n";
    int64 e10 = getTickCount();
    double timeTD = (e10 - e9)/getTickFrequency();
    cout << "TD calculated in " << floor(timeTD/60) << " min " << fmod(timeTD,60) << " s\n" << endl;
    
    int64 e11 = getTickCount();
    double time = (e11 - e1)/getTickFrequency();
    cout << "\nRec3D successfully completed in " << floor(time/60) << " min " << fmod(time,60) << " s\n" << endl;
    
    return 0;
}
