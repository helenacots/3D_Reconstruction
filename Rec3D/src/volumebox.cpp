//
//  volumebox.cpp
//  Rec3D
//
//  Created by Helena on 13/02/2019.
//  Copyright © 2019 Helena. All rights reserved.
//

#include "volumebox.hpp"

void VolumeBox::Update(){
	
	// Build the 3x8 matrix containing the coordinates of the 8 vertices delimiting the vbox
	coord =(Mat_<float>(3,8) << ox, ox, ox, ox, xmax, xmax, xmax, xmax,
			oy, ymax, oy, ymax, oy, ymax, oy, ymax,
			oz, oz, zmax, zmax, oz, oz, zmax, zmax);
	
	// Get the number of voxels in each dimension
//    nx = ceil((xmax)/voxelSize);
//    ny = ceil((ymax)/voxelSize);
//    nz = ceil((zmax)/voxelSize);
	
	// Get the total number of voxels inside the bbox
	nvoxels = nx*ny*nz;
	
}

VolumeBox::~VolumeBox(){
	
}

VolumeBox::VolumeBox(int Ox, int Oy, int Oz, int Nx, int Ny, int Nz, float voxelsize){
	// Define the volume of voxels
	//nvoxels = voxels;
	voxelSize =  voxelsize;
	ox = Ox;
	oy = Oy;
	oz = Oz;
	xmax = Nx*voxelsize;
	ymax = Ny*voxelsize;
	zmax = Nz*voxelsize;
    nx = Nx;
    ny = Ny;
    nz = Nz;

	//double vboxVolume = (xmax-ox)*(ymax-oy)*(zmax-oz);
	//double voxelVolume = vboxVolume/nvoxels;
	//(voxelSize = cbrt(voxelVolume);
	
//    if (voxelSize < 0.005) voxelSize = 0.005;
	
	Update();
	
}

Mat VolumeBox::computeVoxelProjections(vector<Mat> mP, vector< vector<cv::Mat> > imagesC, vector< vector<cv::Mat> > silC, int a, int b ){
	
//	int N = mP.size();
    int sizes[] = {nx,ny,nz};

	Mat voxels_projected_volume(3,sizes,CV_32F);
	
//	int image_counter;
//	int sil_counter;
//	int imgId;
	
	/*int x,y,z, xp, yp, zp;
	Mat currentSil;
#pragma omp parallel for collapse(3) num_threads(8) private(x,y,z, image_counter, sil_counter, currentSil)
	for (int z=0; z < nz; z++) {
		for (int y=0; y < ny; y++) {
			for (int x=0; x < nx; x++) {
				
				// Current voxel index (from 3D array to 1D array)
				//int ind = x + nx * (y + ny * z); // no sé per a què ens pot servir
				
				// Current voxel coordinates in the 3D space
				float xcoord = x*voxelSize + ox + voxelSize/2;
				float ycoord = y*voxelSize + oy + voxelSize/2;
				float zcoord = z*voxelSize + oz + voxelSize/2;
				
				// De moment nomes per la primera imatge de cada camera (estatic)
				image_counter = 0;
				sil_counter = 0;
			#pragma omp parallel for private(imgId, xp, yp, zp)
				for (int camId=0; camId < N; camId++) {
					imgId = 0;
					currentSil = silC[camId][imgId];
					int w = imagesC[camId][imgId].cols;
					int h = imagesC[camId][imgId].rows;
					// Project the voxel from the 3D space to the images
					Mat P = mP[camId];
					Mat projection = P*(Mat_<float>(4,1) << xcoord,ycoord,zcoord,1.0);
					//We get the point in homog coord.
					float xp = projection.at<float>(0);
					float yp = projection.at<float>(1);
					float zp = projection.at<float>(2);
					// Get the cartesian coord
					xp = cvRound(xp/zp);
					yp = cvRound(yp/zp);
					
					// Si voxel dins de imagesC[camId][imgId] --> image_counter++;
					// Si voxel dins de silC[camId][imgId] --> sil_counter++;
					if(xp >= 0 && xp <= w && yp >= 0 && yp <= h){
						image_counter++;
						currentSil.convertTo(currentSil, CV_32F);
						int value = currentSil.at<uchar>(xp,yp);
//						float valueint = value;

//						currentSil.convertTo(currentSil, CV_32FC1);
//						if(currentSil.at<double>(xp,yp)){
							if( value == 255){
								//							cout << "xp: " << xp << " yp: " << yp << "\n";
								sil_counter++;
							}
//						}
						
					}
				}
				if(image_counter > a && sil_counter > b) {
					// set voxel to 1
					voxels_projected_volume.at<int>(x,y,z) = 1;
					
				} else {
					// set voxel to 0
					voxels_projected_volume.at<int>(x,y,z) = 0;
				}
			}
		}
	}
	
	toVTK("/Users/helena/Documents/TFG/volume.vtk",voxels_projected_volume);*/
	
	return voxels_projected_volume;
	
}

void VolumeBox::toVTK(const char* filename, cv::Mat m){
	//WRITE to VTK file format .vtk ASCII to visualize it
	
	FILE * fid;
	int x,y,z;
	fid = fopen(filename,"w");
	
	fprintf(fid,"# vtk DataFile Version 2.0\n");
	fprintf(fid,"Volume example\n");
	fprintf(fid,"ASCII\n");
	fprintf(fid,"DATASET STRUCTURED_POINTS\n");
	fprintf(fid,"DIMENSIONS %d %d %d\n",nx,ny,nz);
	fprintf(fid,"ASPECT_RATIO %d %d %d\n",1,1,1);
	fprintf(fid,"ORIGIN %f %f %f\n",ox,oy,oz);
	fprintf(fid,"POINT_DATA %d\n",nvoxels);
	fprintf(fid,"SCALARS VolumeBox char 1\n");
	fprintf(fid,"LOOKUP_TABLE default\n");
#pragma omp parallel for num_threads(8) private(z) ordered
	for(int z = 0; z<nz; z++){
	#pragma omp parallel for num_threads(8) private(y) ordered
		for(int y = 0; y<ny; y++){
		#pragma omp parallel for num_threads(8) private(x) ordered
			for(int x = 0; x<nx; x++){
				#pragma omp ordered
				fprintf(fid, "%d ",m.at<int>(x,y,z));
			}
		}
		fprintf(fid, "\n");
	}
	
	fclose(fid);
}

void VolumeBox::toVTK_float(const char* filename, cv::Mat m){
	//WRITE to VTK file format .vtk ASCII to visualize it
	
	FILE * fid_f;
	int x,y,z;
	fid_f = fopen(filename,"w");
	
	fprintf(fid_f,"# vtk DataFile Version 2.0\n");
	fprintf(fid_f,"Volume example\n");
	fprintf(fid_f,"ASCII\n");
	fprintf(fid_f,"DATASET STRUCTURED_POINTS\n");
	fprintf(fid_f,"DIMENSIONS %d %d %d\n",nx,ny,nz);
	fprintf(fid_f,"ASPECT_RATIO %d %d %d\n",1,1,1);
	fprintf(fid_f,"ORIGIN %f %f %f\n",ox,oy,oz);
	fprintf(fid_f,"POINT_DATA %d\n",nvoxels);
	fprintf(fid_f,"SCALARS VolumeBox float 1\n");
	fprintf(fid_f,"LOOKUP_TABLE default\n");
#pragma omp parallel for num_threads(8) private(z) ordered
	for(int z = 0; z<nz; z++){
	#pragma omp parallel for num_threads(8) private(y) ordered
		for(int y = 0; y<ny; y++){
		#pragma omp parallel for num_threads(8) private(x) ordered
			for(int x = 0; x<nx; x++){
			    #pragma omp ordered
				fprintf(fid_f, "%f ",m.at<float>(x,y,z));
			}
		}
		fprintf(fid_f, "\n");
	}
	
	fclose(fid_f);
}

void VolumeBox::store_depth_image(const char* filename, cv::Mat m, int npixels, int h, int w){
    //WRITE to VTK file format .vtk ASCII to visualize it
    
    FILE * fid_f;
    fid_f = fopen(filename,"w");
    int width = w;
    int height = h;
    int col, row;
    fprintf(fid_f,"# vtk DataFile Version 2.0\n");
    fprintf(fid_f,"Depth image\n");
    fprintf(fid_f,"ASCII\n");
    fprintf(fid_f,"DATASET STRUCTURED_POINTS\n");
    fprintf(fid_f,"DIMENSIONS %d %d\n",h,w);
    fprintf(fid_f,"ASPECT_RATIO %d %d\n",1,1);
    fprintf(fid_f,"POINT_DATA %d\n",npixels);
    fprintf(fid_f,"SCALARS Image float 1\n");
    fprintf(fid_f,"LOOKUP_TABLE default\n");
//#pragma omp parallel for num_threads(8) private(col) ordered
    for(int col = 0; col < width; col++){
//    #pragma omp parallel for num_threads(8) private(row) ordered
        for(int row = 0; row < height; row++){
//            #pragma omp ordered
            fprintf(fid_f, "%f ",m.at<float>(row, col));
        }
//    #pragma omp ordered
    fprintf(fid_f, "\n");
    }
    fclose(fid_f);
}

bool VolumeBox::compareMax(float a, float b)
{
    b = 0.3; // mu
    return (a < b); // compare distances to camera (stored in coordinate 'y')
}

void VolumeBox::writeIsoSurfaceVTK(const char* filename, Mat TD, vector<float> TD_array){
    
    double maxvalue, minvalue;
    maxvalue = *max_element(begin(TD_array), end(TD_array));
    minvalue = *min_element(begin(TD_array), end(TD_array));
    cout << "maxvalue = " << maxvalue << ", minvalue = " << minvalue << endl;
//    double maxi;
//    maxi = *max_element(begin(TD_array), end(TD_array), compareMax);

    //cout << "maximum value < 0.3(fora) = " << maxi << endl;
    
    FILE * fid_f;
    int x,y,z;
    fid_f = fopen(filename,"w");
    
    fprintf(fid_f,"# vtk DataFile Version 2.0\n");
    fprintf(fid_f,"TD isosurface\n");
    fprintf(fid_f,"ASCII\n");
    fprintf(fid_f,"DATASET STRUCTURED_POINTS\n");
    fprintf(fid_f,"DIMENSIONS %d %d %d\n",nx,ny,nz);
    fprintf(fid_f,"SPACING %d %d %d\n",1,1,1);
    fprintf(fid_f,"ORIGIN %f %f %f\n",ox,oy,oz);
    fprintf(fid_f,"POINT_DATA %d\n",nx*ny*nz);
    fprintf(fid_f,"SCALARS scalars float 1\n");
    fprintf(fid_f,"LOOKUP_TABLE default\n");
#pragma omp parallel for num_threads(8) private(z) ordered
    for(int z = 0; z<nz; z++){
    #pragma omp parallel for num_threads(8) private(y) ordered
        for(int y = 0; y<ny; y++){
        #pragma omp parallel for num_threads(8) private(x) ordered
            for(int x = 0; x<nx; x++){
                #pragma omp ordered
                fprintf(fid_f, "%f ", 255 * (TD.at<float>(x,y,z) - minvalue) / (maxvalue - minvalue));
            }
        }
        fprintf(fid_f, "\n");
    }
    
    fclose(fid_f);
    
    
//    float minvalue = 0;
//    float maxvalue = 1;
    
    //ofstream fout(filename.c_str());

    //fout << "# vtk DataFile Version 2.0\n";
    //fout << "Probability Volume\n";
    //fout << "ASCII\n";
    //fout << "\n";
    //fout << "DATASET STRUCTURED_POINTS\n";
    //fout << "DIMENSIONS " << nx << " "
    //<< ny << " "
    //<< nz << "\n";
    //fout << "ORIGIN 0 0 0\n";
    //fout << "SPACING 1 1 1\n";
    //fout << "POINT_DATA " << nx * ny * nz << "\n";
    //fout << "SCALARS scalars unsigned_char 1\n";
    //fout << "LOOKUP_TABLE default\n";
    /*
    for (int n=0; n < TD.size(); n++){
        int v = int(255 * (TD[n] - minvalue) / (maxvalue - minvalue));
        fout << std::max(0, std::min(v, 255)) << "\n";
    }*/
}

/*	Aquesta funcció crea un vector de 8 posicions (8 càmeres) que conté per a cada un dels voxels del volum V la distància (depth)
 	des d'aquest voxel fins al centre de la càmera corresponent. 		*/
vector<Mat> VolumeBox::create_distance_lut(const vector<Mat> &centers, const vector<float> &idxs_goodvoxels)
{
	//Point3D worldPoint;
	float halfStep = (float) 0.5*voxelSize; // first voxel center at (halfCubeSize, halfCubeSize, halfCubeSize) instead of (0,0,0)
	int sizes[] = {nx,ny,nz};

	float fOffsetX =  ox;
	float fOffsetY =  oy;
	float fOffsetZ =  oz;
    float cameraX, cameraY, cameraZ;
//    unsigned int indexCam;
    int x,y,z;
	
	//distance LookUpTable
	vector<Mat> distanceLUT(centers.size(),Mat::zeros(3, sizes, CV_32F));
#pragma omp parallel for private (cameraX,cameraY,cameraZ) // i indexCam no es private??
	for(int indexCam = 0; indexCam < centers.size(); indexCam++){
        // aqui he de tenir en compte que les coordenades dels centres estan al reves!
		float cameraX = centers[indexCam].at<float>(0);
		float cameraY = centers[indexCam].at<float>(1);
		float cameraZ = centers[indexCam].at<float>(2);
		//pdata = distanceLUT(indexCam).data();
        Mat_<float> auxMat = Mat::zeros(3, sizes, CV_32FC1);
    #pragma omp parallel for collapse(3) private(x,y,z)
        for (int z = 0; z < nz; z++){
            for (int y = 0; y < ny; y++){
				for (int x = 0; x < nx; x++)
				{
					float idx = (float) x + nx * (y + ny * z); // index del voxel passant de array 3D a array 1D
                    float worldPoint[3];
					// PER TOTS ELS VOXELS  on V(x)=1 --> comprovem si l'index està dins de l'index dels voxels bons
					if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()){ // if it is true, then:
						/* Important: the way the world coordinates are computed MUST be the same in every LUT */
						worldPoint[0] = x * voxelSize + fOffsetX + halfStep - cameraX;
						worldPoint[1] = y * voxelSize + fOffsetY + halfStep - cameraY; // - cameraY
						worldPoint[2] = z * voxelSize + fOffsetZ + halfStep - cameraZ;
						float distance = sqrt(worldPoint[0] * worldPoint[0] + worldPoint[1] * worldPoint[1] + worldPoint[2] * worldPoint[2]);
//						cout << distance << "      ";
//                        distanceLUT[indexCam].at<float>(x,y,z) = distance;
                        auxMat.at<float>(x,y,z) = distance;
					}
				}
            }
        }
        distanceLUT[indexCam] = auxMat;
        auxMat.release();
        
	}
	return distanceLUT;
}

/*	1. Per cada imatge pertanyen a cada una de les càmeres, mirar del seu volum de voxels V, mirem en el vector voxel_luts que guarda les coordenades del voxel projectat a la imatge corresponent (xp2d,yp2d) , mirar si aquestes coordenades es troben dins dels limits correctes llavors calcular la distància del voxel corresponent a la camera (a partir del vector distanceLut) i guardar l'index del voxel juntament amb aquesta distància. Aquests valors els guardem a l'array lutsRay.
 
 	2. Ordenem el vector lutsRay de forma que per cada pixel de la imatge tenim els voxels pels quals el seu raig passa ordenats de menor a major distància.
 
 */

// comparar les distàncies (depths) de dos voxels per ordenar-los de menor a major distància
bool VolumeBox::compare(const Mat& a, const Mat& b)
{
	if(!a.data || !b.data){
		return {};
	}else if(a.empty() || b.empty()){
		return {};
	}else{
//		cout << "a(1): " << a.at<float>(1) << ", b(1): " << b.at<float>(1) << "\n";
		return (a.at<float>(1) < b.at<float>(1)); // compare distances to camera (stored in coordinate 'y')
	}
}

void VolumeBox::sort_voxels_inRay_by_distance(vector< vector< vector< vector<Mat> > > > &lutsRay_aux, const vector< vector< vector< vector<Mat> > > > &voxel_luts, const vector<Mat> &distances, const int width, const int height, const vector<float> &idxs_goodvoxels){
	
//    unsigned int indexCam,
//    int Nc=lutsRay_aux.size();
//    unsigned int x, y, z;
//    float idx;
//
//    // 1. create ray lut for each image and pixel p
//#pragma omp parallel for private(x,y,z,idx)
//    for (int indexCam=0; indexCam<Nc; indexCam++){ // for each image
//        for(int z=0; z<nz; z++){
//            for(int y=0; y<ny; y++)
//                for(int x=0; x<nx; x++)
//                {
//                    float idx = (float) x + nx * (y + ny * z); // index del voxel passant de array 3D a array 1D
//                    // PER TOTS ELS VOXELS  on V(x)=1 --> comprovem si l'index està dins de l'index dels voxels bons
//                    if(find(idxs_goodvoxels.begin(),idxs_goodvoxels.end(),idx) != idxs_goodvoxels.end()){ // if it is true, then:
//                        int xp2d = voxel_luts[indexCam][x][y][z].at<int>(0); // columna (c)
//                        int yp2d = voxel_luts[indexCam][x][y][z].at<int>(1); // fila (r)
//                        // calculem les distàncies per aquells pixels als que projecten a voxels, sino no cal
////                        cout << "xp2d: " << xp2d << ", yp2d: " << yp2d << "\n";
//                        Mat tmp(2,1,CV_32FC1);
//                        tmp.at<float>(0) = idx;
//                        tmp.at<float>(1) = distances[indexCam].at<float>(x,y,z); //distanceLut vector --> get the distance of the current voxel
////                        cout << tmp << "\n";
//
//                        // afegir l'index i la distancia del voxel al pixel corresponent.
//                        lutsRay_aux[indexCam][yp2d][xp2d].push_back(tmp);
////                        tmp.copyTo(lutsRay_aux[indexCam][xp2d][yp2d]);
//                    }
//
//                }
//            }
//        }
//
//    // 2. sort each pixel ray according to the distance from voxels to the camera
//#pragma omp parallel for
//    for (int indexCam=0; indexCam<Nc; indexCam++){
//        // recorre els pixels de la imatge i agafar el raig que passa per les coordenades de tal pixel
//        for(int r=0; r<height; r++){ // cada 4 pixels per anar més rapid
//            for(int c=0; c<width; c++)
//            {
//                int n_voxels = lutsRay_aux[indexCam][r][c].size();
////                cout << "nvoxels on the ray: " << n_voxels << "\n";
//                if(n_voxels != 0){
////                    for (int f = 0; f < n_voxels; f++) {
////                        cout << lutsRay_aux[indexCam][r][c][f] << "\n"; // podem veure que està ben guardat les diferents matrius tmp
////                    }
////                    for (auto i = lutsRay_aux[indexCam][r][c].begin(); i != lutsRay_aux[indexCam][r][c].end(); ++i)
////                        std::cout << *i << '\n';
//                    if(n_voxels >4){
//                        cout << "wtf voxels!!!!    ";
//                    }
//                    sort(lutsRay_aux[indexCam][r][c].begin(),lutsRay_aux[indexCam][r][c].end(), compare);
//
////                    for (auto i = lutsRay_aux[indexCam][r][c].begin(); i != lutsRay_aux[indexCam][r][c].end(); ++i)
////                        std::cout << *i << '\n';
//                }
//
//            }
//        }
//    }
}

	

	
