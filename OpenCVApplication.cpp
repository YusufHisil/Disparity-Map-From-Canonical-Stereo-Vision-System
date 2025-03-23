#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#define DISPARITY_MIN 0
#define DISPARITY_MAX 63

using namespace std;


Mat calculateDisparityMaps(char* left_image, char* right_image, short int block_size);
void mouseClickCallback(int event, int x, int y, int flags, void* param);
void readCameraParameters(const char* pname,float* f,float center[2],float Lcam[3],float Rcam[3],float rot[3][3]);
void showParameters();


// focal length
float f=10;
// optical center
float center[2];
// left and right camera position and baseline;
float Lcam[3], Rcam[3], B;
// camera rotation matrix
float rot[3][3];

void main()
{
	// surpress opencv logging
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

	// folder path
	char fname[MAX_PATH];
	// left photo path
	char lname[MAX_PATH];
	// right photo path
	char rname[MAX_PATH];
	// parameters path
	char pname[MAX_PATH];

	
	while (openFolderDlg(fname))
	{
		strcpy(lname, fname);
		strcat(lname, "\\left.bmp");
		strcpy(rname, fname);
		strcat(rname, "\\right.bmp");
		strcpy(pname, fname);
		strcat(pname, "\\parameters.txt");

		readCameraParameters(pname, &f, center, Lcam, Rcam, rot);
		showParameters();

		Mat left = imread(lname, IMREAD_GRAYSCALE);
		imshow("left", left);

		Mat res5x5 = calculateDisparityMaps(lname, rname, 5);
		Mat res7x7 = calculateDisparityMaps(lname, rname, 7);
		Mat res9x9 = calculateDisparityMaps(lname, rname, 9);

		imshow("5x5", res5x5 * 4);//MULTIPLYING BY 4 FOR BETTER IMAGE BRIGHTNESS
		imshow("7x7", res7x7 * 4);
		imshow("9x9", res9x9 * 4);

		setMouseCallback("5x5", mouseClickCallback, &res5x5);
		setMouseCallback("7x7", mouseClickCallback, &res7x7);
		setMouseCallback("9x9", mouseClickCallback, &res9x9);

		waitKey();
		destroyAllWindows();
	}
}


// calculates both disparity maps for the given window=block=matrix size
Mat calculateDisparityMaps(char* left_image, char* right_image, short int block_size)
{
	Mat left = imread(left_image, IMREAD_GRAYSCALE);
	Mat right = imread(right_image, IMREAD_GRAYSCALE);

	uchar* lpLeft = left.data;
	uchar* lpRight = right.data;

	//image height
	int h = left.rows;
	//image width
	int w = left.cols;
	
	// left relative disparity --> search in the right image along the epipolar line to the left of the reference coordinates
	Mat l_r_d= Mat(h, w, CV_8UC1);
	uchar* lp_lrd = l_r_d.data;
	
	int step = (int)left.step;

	// cost function result - sum of absolute differences
	int SAD;
	int min_SAD;
	// disparity for minimum SAD
	int opt_d = 0;

	// for every pixel in the images
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{	
			min_SAD = INT_MAX;
			// for the line defined by the disparity interval [0,63]
			for(int d = DISPARITY_MIN; d < DISPARITY_MAX; d++)
			{
				if ((j - d) < 0) break;
				SAD = 0;
				// for each pixel in the window
				for (int y = i - block_size / 2; y <= i + block_size / 2; y++)
				{
					if (y < 0) continue;
					else if (y >= h) break;

					for (int x = j - block_size / 2; x <= j + block_size / 2; x++)
					{
						if (x - d < 0) continue;
						else if (x >= w) break;

						SAD += abs(lpLeft[y * step + x] - lpRight[y * step + x - d]);
						//SAD += abs(left.at<uchar>(y, x) - right.at<uchar>(y, x - d));
					}
				}

				if (SAD < min_SAD)
				{
					min_SAD = SAD;
					opt_d = d;
				}
			}
			lp_lrd[i * step + j] = opt_d;
			//l_r_d.at<uchar>(i, j) = opt_d;
		}
	}
	
	// disparity relative to the right image
	Mat r_r_d= Mat(h, w, CV_8UC1);
	uchar* lp_rrd = r_r_d.data;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			min_SAD = INT_MAX;
			// for the line defined by the disparity interval [0,63]
			for (int d = DISPARITY_MIN; d < DISPARITY_MAX; d++)
			{
				if ((j + d) >= w) break;
				SAD = 0;
				// for each pixel in the window
				for (int y = i - block_size / 2; y <= i + block_size / 2; y++)
				{
					if (y < 0) continue;
					else if (y >= h) break;

					for (int x = j - block_size / 2; x <= j + block_size / 2; x++)
					{
						if (x < 0) continue;
						else if (x + d >= w) break;

						//SAD += abs(right.at<uchar>(y, x) - left.at<uchar>(y, x + d));
						SAD += abs(lpRight[y * step + x] - lpLeft[y * step + x + d]);
					}
				}

				if (SAD < min_SAD)
				{
					min_SAD = SAD;
					opt_d = d;
				}
			}
			//r_r_d.at<uchar>(i, j) = opt_d;
			lp_rrd[i * step + j] = opt_d;
		}
	}


	// final disparity map 
	// takes values from the map calculated relative to the left image
	// with the added check of disparity equivalence between equivalent pixels of the calculated disparity maps
	// in other words, if the two values of each respective pairs of pixels are not equal, the algorithm sets them to 0
	// to mark them as "no depth data", because the computation yielded inadequate results (no final depth data)
	// possible causes: 
	Mat f_d = Mat(h, w, CV_8UC1);
	uchar* lp_fd = f_d.data;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			//uchar d = l_r_d.at<uchar>(i, j);
			uchar d = lp_lrd[i * step + j];
			if (j - d < 0)
			{
				//f_d.at<uchar>(i, j) = 0; // this makes sure that the points that don't appear because of edge disparity are always black
				lp_fd[i * step + j] = 0;
				continue;
			}

			if (d != lp_rrd[i * step + j - d]/*r_r_d.at<uchar>(i, j - d)*/)
				//f_d.at<uchar>(i, j) = 0;
				lp_fd[i * step + j] = 0;
			else
				//f_d.at<uchar>(i, j) = d;
				lp_fd[i * step + j] = d;
		}
	}
	return f_d;
}

//!TODO: review the formulas, add crosses on click points for clarity
// calculates the real world coordinates for any valid selected point
void mouseClickCallback(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat src = *((Mat*)param);
		uchar d = src.at<uchar>(y, x);
		if (d == 0) printf("no depth data\n");
		else
		{
			float Z = f * B / (float)d;
			float X = (x - center[0]) * Z / f;
			float Y = (y - center[1]) * Z / f;

			printf("2D(%d, %d) d=%d => 3D(%.2f, %.2f, %.2f)\n", x, y, d, X, Y, Z);
		}
	}
}

void readCameraParameters(const char* pname,
	float *f,
	float center[2],
	float Lcam[3],
	float Rcam[3],
	float rot[3][3])
{
	ifstream file(pname);
	if (!file.is_open()) {
		cerr << "Failed to open file: " << pname << endl;
		return;
	}

	string line;
	// Skip "Intrinsic parameters values:"
	getline(file, line);

	// Read focal length
	getline(file, line);
	if (line.find("f =") != string::npos) {
		string value_str = line.substr(line.find('=') + 1);
		*f = stof(value_str);
	}
	// Read optical center xC and yC
	getline(file, line);
	sscanf(line.c_str(), "xC = %f", &center[0]);
	getline(file, line);
	sscanf(line.c_str(), "yC = %f", &center[1]);

	// Skip "Left camera position:"
	getline(file, line);

	// Read left camera position
	getline(file, line);
	sscanf(line.c_str(), "XCl = %f", &Lcam[0]);
	getline(file, line);
	sscanf(line.c_str(), "YCl = %f", &Lcam[1]);
	getline(file, line);
	sscanf(line.c_str(), "ZCl = %f", &Lcam[2]);

	// Skip "Right camera position:"
	getline(file, line);

	// Read right camera position
	getline(file, line);
	sscanf(line.c_str(), "XCr = %f", &Rcam[0]);
	getline(file, line);
	sscanf(line.c_str(), "YCr = %f", &Rcam[1]);
	getline(file, line);
	sscanf(line.c_str(), "ZCr = %f", &Rcam[2]);


	B = 0;
	for (int i = 0; i < 3; i++) B += pow(Lcam[i] - Rcam[i], 2);
	B = sqrt(B);

	// Skip "Camera rotation:"
	getline(file, line);

	// Read 3 lines of rotation matrix
	for (int i = 0; i < 3; ++i) {
		getline(file, line);
		istringstream iss(line);
		iss >> rot[i][0] >> rot[i][1] >> rot[i][2];
	}

	file.close();
}

void showParameters()
{
	printf("Camera intrinsic and extrinsic parameters: \n");
	printf("f= %f\n", f);
	printf("center point: \n");
	for (int i = 0; i < 2; i++)
		printf("%f, ", center[i]);
	puts("\nLeft camera position: ");
	for (int i = 0; i < 3; i++)
		printf("%f, ", Lcam[i]);
	puts("\nRight camera position: ");
	for (int i = 0; i < 3; i++)
		printf("%f, ", Rcam[i]);
	puts("\nWorld rotation matrix for both cameras:\n");
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			printf("%f, ", rot[i][j]);
		printf("\n");
	}
}