#pragma once

#ifndef MapHead
#define MapHead

//Standard Libraries
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"

//OpenCV Libraries
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

//Custom Libraries
#include "MatHeader.h"
#include "Segmentation.h"

using namespace cv;
using namespace std;

#define BASE_MAP_SIZE 20000

class ThreeD_Map {
protected:
	//Matrix object
	Image_Process map_details;
	short basic_container[BASE_MAP_SIZE][BASE_MAP_SIZE][BASE_MAP_SIZE] = { 0 };

private:

public:
	void import_matrix(Image_Process container) { map_details = container; }

	void initialise_base_map()
	{
		for (short obj = 0; obj <= map_details.return_obj_count(); obj++)
		{
			
		}
	}

};


#endif