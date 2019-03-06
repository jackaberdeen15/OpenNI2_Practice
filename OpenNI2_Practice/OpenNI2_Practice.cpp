// OpenNI2_Practice.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//General Headers
#include <stdio.h>
#include "pch.h"
#include <iostream>
#include "MatHeader.h"
#include "Segmentation.h"

//OpenNi2 Headers
#include <OpenNI.h>

//NiTE2 Headers
#include <NiTE.h>

//GLUT headers
//#include<GL/glut.h>

//opencv2 headers
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"


//////////////////////// Declerations /////////////

using namespace std;
using namespace cv;
using namespace openni;

int window_w = 640; //window x size of texture for glut
int window_h = 480; // window y size of texture for glut
//int elapsed_time = 0; //integer for time passed
//OniRGB888Pixel* gl_texture; //type of texture buffer(3 bytes, red, blue and green)

VideoStream depthSensor, colourSensor; //object for the video stream
VideoFrameRef irf, rgb;    //IR and RGB VideoFrame Class Object
VideoMode Dvmode,Cvmode;      // VideoMode Object
Device device; //object for the kinect device
Image_Process Depth_Proc,Colour_Proc; //object for the matrix which will contain average depths and block labels
#define KINECT_AVAIL 1



////////////////////////  Start of Code ///////////

char ReadLastCharOfLine()
{
	int newChar = 0;
	int lastChar;
	fflush(stdout);
	do
	{
		lastChar = newChar;
		newChar = getchar();
	} while ((newChar != '\n') && (newChar != EOF));
	return (char)lastChar;
}

int GetDevice() //function to get number of devices connected and information about them
{
	//code to get list of devices connected and some basic info about them (not needed rn)

	openni::Array<openni::DeviceInfo> listOfDevices; //custom array type used by openni
	openni::OpenNI::enumerateDevices(&listOfDevices);

	int numberOfDevices = listOfDevices.getSize();

	if (numberOfDevices > 0) {
		printf("%d Device(s) are available to use. \r\n\r\n", numberOfDevices);

		for (int i = 0; i < numberOfDevices; i++)
		{
			openni::DeviceInfo device = listOfDevices[i];
			printf("%d. %s => %s (VID: %d | PID: %d) is connected at %s\r\n", i, device.getVendor(), device.getName(), device.getUsbVendorId(), device.getUsbProductId(), device.getUri());
		}
	}
	else {
		printf("No device connected to this machine.");
	}

	int selected = 0;
	do
	{
		printf("Select your desired device and then press Enter to continue.\r\n");
		selected = ReadLastCharOfLine() - 'a';
	} while (selected < 0 || selected >= numberOfDevices);

	return selected;
}

//function to handle openni::Status
//will help reduce the number of conditions in our code 
//and thus make it more readable and understandable

bool HandleStatus(Status status)
{
	if (status == STATUS_OK)
		return true;
	printf("ERROR: #%d, %s", status, OpenNI::getExtendedError());
	ReadLastCharOfLine();
	return false;
}

//Possible Status'
//STATUS_OK no problem or error
//STATUS_ERROR error has occured duing execution
//STATUS_NOT_IMPLEMENTED function called or used not implemented yet
//STATUS_NOT_SUPPORTED requested task is not supported or possible
//STATUS_BAD_PARAMETER parameter of method or function is incorrect, null, or irrelevant
//STATUS_OUT_OF_FLOW overflow means problem with memory/device or stack
//STATUS_NO_DEVICE no device connected and available to use

Mat Depth_Func() {
	
	Status status;

	printf("Setting video mode to 640x480x30 Depth 1mm ...\r\n");
	//VideoMode vmod;
	Dvmode.setFps(30);
	Dvmode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	Dvmode.setResolution(window_w, window_h);
	status = depthSensor.setVideoMode(Dvmode);
	if (!HandleStatus(status));
	printf("Done.\r\n");

	printf("Starting Stream ...\r\n");
	status = depthSensor.start();
	if (!HandleStatus(status));
	printf("Done.\r\n");

	int frame_count = 0; //for grabbing pic
	bool should_stop = false; //for grabbing pic
	int colour_frame_count = 0; //for grabbing pic


	Mat frame,image;              // OpenCV Matrix Object, also used to store images
	int h, w;               // Height and Width of the IR VideoFrame

	//Mat frame1;              // OpenCV Matrix Object, also used to store images
	//int h1, w1;              // Height and Width of the IR VideoFrame

	if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		printf("Getting data from video frame...\r\n");
		status = depthSensor.readFrame(&irf);// Read one depth VideoFrame at a time
		if (!HandleStatus(status));
		printf("Done.\r\n");

		printf("Checking if videoframe is valid...\r\n");
		if (irf.isValid())// If the depth and colour VideoFrames are valid
		{
			printf("Videoframe valid.\r\n");
			// Get the IR VideoMode Info for this video stream.
			// This includes its resolution, fps and stream format.
			Dvmode = depthSensor.getVideoMode();

			// PrimeSense gives the IR stream as 16-bit data output
			const uint16_t* imgBuf = (const uint16_t*)irf.getData(); //?????

			h = irf.getHeight();
			w = irf.getWidth();

			// Create the OpenCV Mat Matrix Class Object
			// to receive the IR VideoFrames
			frame.create(h, w, CV_16U);

			memcpy(frame.data, imgBuf, h*w * sizeof(uint16_t));
			//convert from 16-bit to 8-bit

			printf("Storing frame data to Opencv Matrix Object...\r\n");
			frame.data = ((uchar*)irf.getData());
			printf("Done.\r\n");

			Mat frametodisplay;

			printf("Coverting Matrix data from 16bit to 8bit...\r\n");
			frame.convertTo(frametodisplay, CV_8U, 0.05f); //THIS ONE!!! <3 ????????
			printf("Done.\r\n");

			printf("Storing Data to GRABBED.png...\r\n");
			cv::imwrite("GRABBED.png", frametodisplay);
			printf("Done.\r\n");



			printf("Displaying depth frame on screen...\r\n");

			//read image
			image = imread("GRABBED.png", IMREAD_GRAYSCALE); //stores the image data into opencv matrix object
			//create image window
			namedWindow("Depth Image", WINDOW_AUTOSIZE); // Create a named window
			imshow("Depth Image", image);  // Show the depth VideoFrame in this window
			printf("Done.\r\n");
			waitKey(1);
			
			printf("Press Enter to Continue...\r\n");
			ReadLastCharOfLine();
			destroyAllWindows();
			

			printf("Processing Image...\r\n");
			printf("Segmenting image...\r\n");
			//printf("Press Enter to Continue...\r\n");
			//ReadLastCharOfLine();

			Depth_Proc.setup_block_matrix(640, 480);
			image = Depth_Proc.Basic_Segment(image);

			printf("Grouping image...\r\n");
			//printf("Press Enter to Continue...\r\n");
			//ReadLastCharOfLine();

			Depth_Proc.Grouping(image);
			printf("Done.\r\n");

			//printf("Press Enter to Continue...\r\n");
			//ReadLastCharOfLine();

			printf("Displaying Processed Image.\r\n");
			namedWindow("Processed Image", WINDOW_AUTOSIZE); // Create a named window
			imshow("Processed Image", image);  // Show the depth VideoFrame in this window
			printf("Done.\r\n");
			waitKey(1);
		}
	}

	printf("Stage completed. Press ENTER to Continue...\r\n");
	ReadLastCharOfLine();
	destroyAllWindows();
	return image;

}

Mat Colour_Func(){

	Status status;

	printf("Setting video mode to 640x480x30 Colour RGB888 ...\r\n");
	//VideoMode vmod;
	Cvmode.setFps(30);
	Cvmode.setPixelFormat(PIXEL_FORMAT_RGB888);
	Cvmode.setResolution(window_w, window_h);
	status = colourSensor.setVideoMode(Cvmode);
	if (!HandleStatus(status));
	printf("Done.\r\n");

	printf("Starting Stream ...\r\n");
	status = colourSensor.start();
	if (!HandleStatus(status));
	printf("Done.\r\n");

	int frame_count = 0; //for grabbing pic
	bool should_stop = false; //for grabbing pic
	int colour_frame_count = 0; //for grabbing pic


	Mat frame,image;              // OpenCV Matrix Object, also used to store images
	int h, w;               // Height and Width of the IR VideoFrame

	//Mat frame1;              // OpenCV Matrix Object, also used to store images
	//int h1, w1;              // Height and Width of the IR VideoFrame

	if (device.getSensorInfo(SENSOR_COLOR) != NULL)
	{
		printf("Getting data from video frame...\r\n");
		status = colourSensor.readFrame(&rgb);// Read one depth VideoFrame at a time
		if (!HandleStatus(status));
		printf("Done.\r\n");

		printf("Checking if videoframe is valid...\r\n");
		if (rgb.isValid())// If the depth and colour VideoFrames are valid
		{
			printf("Videoframe valid.\r\n");
			// Get the Colour VideoMode Info for this video stream.
			// This includes its resolution, fps and stream format.
			Cvmode = colourSensor.getVideoMode();

			// PrimeSense gives the Colour stream as 24-bit data output
			const RGB888Pixel* imgBuf = (const RGB888Pixel*)rgb.getData(); 

			h = rgb.getHeight();
			w = rgb.getWidth();

			printf("Height of stream: %d, width of stream: %d.\r\n", h, w);
			

			// Create the OpenCV Mat Matrix Class Object
			// to receive the Colour VideoFrames
			frame.create(h, w, CV_8UC3);

			memcpy(frame.data, imgBuf, h * w * 3 * sizeof(uint8_t));

			printf("Storing Data to ColourGrab.png...\r\n");
			cv::cvtColor(frame, frame, CV_BGR2RGB); //this will put colors right
			cv::imwrite("ColourGrab.png", frame);
			printf("Done.\r\n");


			printf("Displaying colour frame on screen...\r\n");

			//read image
			Mat image = imread("ColourGrab.png", IMREAD_COLOR); //stores the image data into opencv matrix object
			
			//create image window
			namedWindow("Colour Image", WINDOW_AUTOSIZE); // Create a named window
			imshow("Colour Image", image);  // Show the colour VideoFrame in this window
			printf("Done.\r\n");
			waitKey(1);
			printf("Press Enter to Continue...\r\n");
			ReadLastCharOfLine();
			destroyAllWindows();
		}
	}

	printf("Stage completed. Press ENTER to terminate...\r\n");
	ReadLastCharOfLine();
	destroyAllWindows();

	return image;
}

Mat Detect_Colour(){

	Mat Image = imread("ColourGrab.png", IMREAD_COLOR);
	Mat Mask;
	short min = 250;
	//cv::cvtColor(Image, Image, COLOR_BGR2RGB);

	printf("Filtering colours...\r\n");
	Scalar lowerb = cv::Scalar(min, min, min);
	Scalar upperb = cv::Scalar(255, 255, 255);
	inRange(Image, lowerb, upperb, Mask);
	printf("Done.\r\n");

	printf("Smoothing image...\r\n");
	//addWeighted(lowerb, 1.0, upperb, 1.0, 0.0, Mask);
	blur(Mask, Mask, Size(5, 5), Point(-1, -1), 4);
	erode(Mask, Mask, Mat(), Point(-1, -1));
	dilate(Mask, Mask, Mat(), Point(-1, -1), 1, 1, 1);

	printf("Done.\r\n");

	//namedWindow("Masked Image", WINDOW_AUTOSIZE); // Create a named window

	imshow("Mask", Mask);
	waitKey(1);
	//cvtColor(Mask, Mask, COLOR_RGB2GRAY);
	imwrite("ColourFiltered.png", Mask);


	printf("Press Enter to Continue...\r\n");
	ReadLastCharOfLine();
	destroyAllWindows();

	return Mask;
}

Point* Detect_Rectangles() {

	Mat filt_image = imread("ColourFiltered.png");
	Mat orig_image = imread("ColourGrab.png");
	//cvtColor(image, image, COLOR_GRAY2RGB);
	Mat cnymat;
	//Canny edge detection
	Canny(filt_image, cnymat, 75, 200, 3);
	static Point Coords[4];

	//creates a vector of lines populated by houghlines
	vector<Vec2f> lines;
	HoughLines(cnymat, lines, 1, CV_PI / 180, 50, 0, 0);

	printf("%zd lines have been found.\r\n", lines.size());

	if (lines.size() > 3)
	{
		Point C; //where centre of rectange is;
		Point P[500]; //points of line intersection
		double numer3, numer4;
		Point pt1[500], pt2[500]; //output hough line end points
		double m1, m2, m3, m4, m5, m6, denom1, denom2, denom3, denom4, denom5;
		Point f1[500], f2[500]; //arrays for confirmed perpindicular lines f1-->pt1 and f2-->pt2
		int numlines = 0;

		//conversion from polar to cartesian coordinates
		//looping through each line
		printf("Converting from polar to cartesian coordinates...\r\n");
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1[i].x = cvRound(x0 + 1000 * (-b)); //cvRound rounds to nearest int, length of line 1000 is arbitrary
			pt1[i].y = cvRound(y0 + 1000 * a);
			pt2[i].x = cvRound(x0 - 1000 * (-b));
			pt2[i].y = cvRound(y0 - 1000 * a);
			cout << "Line " << i << " Coordinates (" << pt1[i].x << "," << pt1[i].y << ") & (" << pt2[i].x << "," << pt2[i].y << ")" << endl;
		}
		printf("Done.\r\n");
		printf("Placing lines on original image...\r\n");
		for (size_t i = 0; i < lines.size(); i++)
		{
			//put lines on image in cyan
			line(orig_image, pt1[i], pt2[i], Scalar(255,255,0), 2, CV_AA);
		}
		printf("Done.\r\n");

		printf("Checking for perpendicular intersecting lines...\r\n");
		//check for perpendicular lines
		for (size_t i = 0; i < lines.size() - 1; i++)
		{
			for (size_t j = i + 1; j < lines.size(); j++)
			{
				//gradient comparisons between lines
				denom1 = (pt2[i].x - pt1[i].x);
				denom2 = (pt2[j].x - pt1[j].x);

				//allows for all elements to be compared
				if (denom1 == 0) { denom1 = 0.00001; }
				if (denom2 == 0) { denom2 = 0.00001; }

				m1 = (pt2[i].y - pt1[i].y)/denom1;
				m2 = (pt2[j].y - pt1[j].y)/denom2;
				m3 = -(m1 * m2);

				//printf("Denom1 = %f, Denom2 = %f, ", denom1, denom2);
				

				//check if lines intersect at right angle
				if (m3 >= 0.999 && m3 <= 1.001)
				{
					printf("m3 = %f.\r\n", m3);
					//save intersection line coardinates to f1 and f2
					f1[numlines] = pt1[i];
					f2[numlines] = pt2[i];
					numlines++;
					f1[numlines] = pt1[j];
					f2[numlines] = pt2[j];
					numlines++;
				}
			}
		}
		printf("Done.\r\n");
		printf("Numlines = %d.\r\n", numlines);

		
		if (numlines > 3)
		{
			for (size_t i = 0; i < numlines - 1; i++)
			{
				for (size_t j = i + 1; j < numlines; j++) //remove duplicates from array
				{
					if (f1[j].x == f1[i].x && f1[j].y == f1[i].y)
					{
						for (size_t k = j; k < numlines - 1; k++)
						{
							f1[k] = f1[k + 1];
							f2[k] = f2[k + 1];
						}
						numlines--;
						//printf("Numlines is currently %d.\r\n", numlines);
					}
				}
			}
			printf("Duplicates removed. Current numlines = %d.\r\n", numlines);
			printf("Checking intersection points...\r\n");
			//printing lines
			for (size_t i = 0; i < numlines; i++)
			{
				line(orig_image, f1[i], f2[i], Scalar(0, 255, 0), 3, CV_AA);
			}
			int counter = 0;
			//get intersecting lines
			for (size_t i = 0; i < numlines - 1; i++)
			{
				for (size_t j = i + 1; j < numlines; j++)
				{
					denom4 = (f2[i].x - f1[i].x);
					denom5 = (f2[j].x - f1[j].x);
					if (denom4 == 0.0) { denom4 = 0.00001; }
					if (denom5 == 0.0) { denom5 = 0.00001; }
					m4 = (f2[i].y - f1[i].y) / denom1;
					m5 = (f2[j].y - f1[j].y) / denom2;
					m6 = -(m4*m5);
					printf("The Value of m6 is %f.\r\n", m6);

					if (m6 > 0.1 && m6 <= 4)
					{
						cout << "Entered If condition" << endl;
						numer3 = (((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))*(f1[j].x - f2[j].x)) - (((f1[i].x - f2[i].x)*((f1[j].x*f2[j].y) - (f1[j].y*f2[j].x))));
						numer4 = (((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))*(f1[j].y - f2[j].y)) - (((f1[i].y - f2[i].y)*((f1[j].x*f2[j].y) - (f1[j].y*f2[j].x))));
						denom3 = ((f1[i].x - f2[i].x)*(f1[j].y - f2[j].y)) - ((f1[i].y - f2[i].y)*(f1[j].x - f2[j].x));
						short x = (numer3 / denom3);
						short y = (numer4 / denom3);
						cout << "Calc Coordinates are (" << P[counter].x << "," << P[counter].y << ")." << endl;
						if ((x > 0 && x < 640) && (y > 0 && y < 480)) // if within the size of the camera feed (640x480)
						{
							P[counter].x = x;
							P[counter].y = y;
							counter++;	// increment counter
						}
					}
					else
					{
						numer3 = (((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))*(f1[j + 1].x - f2[j + 1].x)) - (((f1[i].x - f2[i].x)*((f1[j + 1].x*f2[j + 1].y) - (f1[j + 1].y*f2[j + 1].x))));
						numer4 = (((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))*(f1[j + 1].y - f2[j + 1].y)) - (((f1[i].y - f2[i].y)*((f1[j + 1].x*f2[j + 1].y) - (f1[j + 1].y*f2[j + 1].x))));
						denom3 = ((f1[i].x - f2[i].x)*(f1[j + 1].y - f2[j + 1].y)) - ((f1[i].y - f2[i].y)*(f1[j + 1].x - f2[j + 1].x));
						short x = (numer3 / denom3);
						short y = (numer4 / denom3);
						if ((x > 0 && x < 640) && (y > 0 && y < 480))			// if within the size of the camera feed (640x480)
						{
							P[counter].x = x;
							P[counter].y = y;
							counter++;												// increment counter
						}
					}
				}
			}
			printf("Final value for counter is %d.\r\n", counter);
			if (counter != 0) {

				printf("Removing duplicates...\r\n");
				for (size_t i = 0; i < counter - 1; i++)
				{
					for (size_t j = i + 1; j < counter; j++)
					{
						printf("i & j = %d,%d. 1st set of coordinates (%d,%d), 2nd set (%d,%d).\r\n", i, j, P[i].x, P[i].y, P[j].x, P[j].y);
						float perc = 0.1;
						if ((P[j].x <= (1 + perc) * P[i].x && P[j].x >= (1 - perc) * P[i].x) && (P[j].y <= (1 + perc) * P[i].y && P[j].y >= (1 - perc) * P[i].y))
						{
							printf("Coordinates within range of eachother.\r\n");
							for (size_t k = j; k < counter - 1; k++)
							{
								P[k] = P[k + 1];
							}
							counter--;
							j += -1;
						}
					}
				}
			}
			printf("New value for counter is %d.\r\n", counter);
			for (size_t i = 0; i < counter; i++)
			{
				printf("Circle placement coordinate is (%d,%d).\r\n", P[i].x, P[i].y);
				circle(orig_image, P[i], 3, Scalar(255, 0, 0), 3, 8, 0);
			}
			for (int i = 0; i < 4; i++) { Coords[i] = P[i]; }//store coordinates of rectangle corners to return;
		}
		imshow("Image", orig_image);
		waitKey(1);

		imwrite("Detected_Image.png", orig_image);

		printf("Press Enter to Continue...\r\n");
		ReadLastCharOfLine();
		destroyAllWindows();


	}

	return Coords;
}

void Detect_Calib()
{
	Mat filt_image = imread("ColourFiltered.png", IMREAD_GRAYSCALE);
	Point P[500]; //store coordinates of centre of Calibration objects

	printf("Processing Filtered Image...\r\n");
	printf("Segmenting Filtered Image...\r\n");
	//printf("Press Enter to Continue...\r\n");
	//ReadLastCharOfLine();

	Colour_Proc.setup_block_matrix(640, 480);
	filt_image = Colour_Proc.Basic_Colour_Segment(filt_image);

	printf("Grouping image...\r\n");
	//printf("Press Enter to Continue...\r\n");
	//ReadLastCharOfLine();

	Colour_Proc.Grouping(filt_image, true);
	printf("Done.\r\n");

	printf("Displaying Processed Image.\r\n");
	namedWindow("Processed Image", WINDOW_AUTOSIZE); // Create a named window
	imshow("Processed Image", filt_image);  // Show the depth VideoFrame in this window
	printf("Done.\r\n");
	waitKey(1);

	printf("Press Enter to Continue...\r\n");
	ReadLastCharOfLine();

}



int main(int argc, char** argv[]) {

	Mat Depth_Image, Colour_Image;

	//ifndef for when the kinect isnt plugged in 
#ifndef KINECT_AVAIL
#define KINECT_AVAIL 1


	Status status = STATUS_OK;
	printf("Scanning machine for devices and loading modules/drivers ...\r\n");

	status = OpenNI::initialize();
	if (!HandleStatus(status)) return 1;
	printf("Completed\r\n");

	printf("Opening first device ...\r\n");
	status = device.open(ANY_DEVICE);
	if (!HandleStatus(status)) return 1;

	printf("%s Opened, Completed.\r\n", device.getDeviceInfo().getName());

	printf("Checking if depth stream is supported ...\r\n");
	if (!device.hasSensor(SENSOR_DEPTH))
	{
		printf("Stream not supported by this device.\r\n");
		return 1;
	}

	printf("Checking if colour stream is supported ...\r\n");
	if (!device.hasSensor(SENSOR_COLOR))
	{
		printf("Stream not supported by this device.\r\n");
		return 1;
	}

	printf("Asking device to create a depth stream ...\r\n");
	status = depthSensor.create(device, SENSOR_DEPTH);
	if (!HandleStatus(status)) return 1;

	printf("Asking device to create a colour stream ...\r\n");
	status = colourSensor.create(device, SENSOR_COLOR);
	if (!HandleStatus(status)) return 1;

#endif // !KINECT_AVAIL

	//Depth_Image = Depth_Func();
	//Colour_Image = Colour_Func();
	Colour_Image = Detect_Colour();
	Detect_Calib();

	//Point Coords1[2];
	Colour_Proc.show_min_max_coords(1);

	Point* max_coords = Colour_Proc.return_max_coords();
	Point* min_coords = Colour_Proc.return_min_coords();
	int col_obj_count = Colour_Proc.return_obj_count();
	

	depthSensor.stop(); //stop the depth sensor stream
	depthSensor.destroy();

	colourSensor.stop(); //stop the colour sensor stream
	colourSensor.destroy();

	return 0;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
