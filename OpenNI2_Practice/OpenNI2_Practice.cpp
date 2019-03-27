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
//#define KINECT_AVAIL 1

float sizeofcalib = 145; //size of the calibration squares in mm


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

Mat Depth_Func(bool printlabels=false) {
	
	Status status;

	printf("Setting video mode to 640x480x30 Depth 1mm ...\r\n");
	//VideoMode vmod;
	Dvmode.setFps(30);
	Dvmode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	Dvmode.setResolution(window_w, window_h);
	status = depthSensor.setVideoMode(Dvmode);
	//if (!HandleStatus(status));
	printf("Done.\r\n");

	printf("Starting Stream ...\r\n");
	status = depthSensor.start();
	//if (!HandleStatus(status));
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
		//HandleStatus(status);
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
			
			printf("Segmenting image...\r\n");
			//printf("Press Enter to Continue...\r\n");
			//ReadLastCharOfLine();

			Depth_Proc.setup_block_matrix(640, 480);
			image = Depth_Proc.Basic_Segment(image);

			printf("Grouping image...\r\n");
			//printf("Press Enter to Continue...\r\n");
			//ReadLastCharOfLine();

			Depth_Proc.Grouping(image,printlabels);
			printf("Done.\r\n");

			cout << "Setting approximate coordinates of objects..." << endl;
			Depth_Proc.set_approx_obj_coords();
			cout << "Done." << endl;

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
	//HandleStatus(status);
	printf("Done.\r\n");

	printf("Starting Stream ...\r\n");
	status = colourSensor.start();
	//HandleStatus(status);
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
		//HandleStatus(status);
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
			printf("Stage completed. Press ENTER to continue...\r\n");
			ReadLastCharOfLine();
		}
	}
	
	destroyAllWindows();

	return image;
}

Mat Detect_Colour(){

	Mat Image = imread("ColourGrab.png", IMREAD_COLOR);
	Mat Mask;
	short min = 250;
	//cv::cvtColor(Image, Image, COLOR_BGR2RGB);

	printf("Filtering colours...\r\n");
	Scalar lowerb = Scalar(0, 0, 60);
	Scalar upperb = Scalar(40, 40, 255);
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

void Detect_Calib(bool printlabels=false)
{
	Mat filt_image = imread("ColourFiltered.png", IMREAD_GRAYSCALE);

	printf("Processing Filtered Image...\r\n");
	printf("Segmenting Filtered Image...\r\n");
	//printf("Press Enter to Continue...\r\n");
	//ReadLastCharOfLine();

	Colour_Proc.setup_block_matrix(640, 480);
	filt_image = Colour_Proc.Basic_Colour_Segment(filt_image);

	printf("Grouping image...\r\n");
	//printf("Press Enter to Continue...\r\n");
	//ReadLastCharOfLine();

	Colour_Proc.Grouping(filt_image, printlabels);
	printf("Done.\r\n");

	cout << "Setting approximate coordinates of objects..." << endl;
	Colour_Proc.set_approx_obj_coords();
	cout << "Done." << endl;

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

	int calib_object_used=0;


	Depth_Image = Depth_Func(true);

	cout << "Do you want to use a calibration object?\r\n1 Yes\r\n2 No" << endl;
	cin >> calib_object_used;

	if (calib_object_used == 1)
	{
		Colour_Image = Colour_Func();
		Colour_Image = Detect_Colour();
		Detect_Calib(true);

		//Point Coords1[2];
		cout << "Colour Process min max coords." << endl;
		Colour_Proc.show_min_max_coords();

		Point* max_coords = Colour_Proc.return_max_coords();
		Point* min_coords = Colour_Proc.return_min_coords();
		int col_obj_count = Colour_Proc.return_obj_count();

		cout << "Adding Calib Points." << endl;
		for (int i = 1; i <= col_obj_count; i++)
		{
			cout << "Object " << i << " Coords are max " << max_coords[i] << ", min" << min_coords[i] << "." << endl;
			Depth_Proc.add_calib_points(min_coords, max_coords, i);
		}

		cout << "Select Object Num to calibrate with: ";
		short obj_num = 0;
		cin >> obj_num;
		//Depth_Proc.print_labels();
		Depth_Proc.determine_scaling(obj_num, sizeofcalib);
	}
	
	short show_coords = 0;
	cout << "Do you want to view the Min and Max Coordinates of all Objects in Depth Frame?\r\n1 Yes\r\n2 No" << endl;
	cin >> show_coords;
	if (show_coords == 1)
	{
		Depth_Proc.get_object_size();
		Depth_Proc.show_min_max_coords();
	}


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
