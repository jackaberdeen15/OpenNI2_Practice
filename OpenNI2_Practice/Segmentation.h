#pragma once
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "MatHeader.h"

#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "math.h"

//define if you want the printout of the distance assigning procedure (to make sure scanning is performed correctly, ......
//....block numbers are assigned correctly and average distances, too
//#define PrintMode
//#define SimpleAssignment
//#define option1
#define option2
#define BLOCKSIZE 5
#define THRESHOLD 6
#define LABEL_ARRAYSIZE 150 //typically depends on how many labels to be stored (depends on the scanning window size)
#define EXP_OBJ_NUM 1000
using namespace cv;
using namespace std;
ofstream fout;

class Image_Process {
protected:
	//  Load the image from file
	bool setup = false;
	Mat LoadedImage;
	//LoadedImage = imread("GRABBED.png", IMREAD_COLOR);
	short objcounter = 0; //by default 1 object (there will always be something)
	Point obj_coords_max[EXP_OBJ_NUM];
	Point obj_coords_min[EXP_OBJ_NUM]; //array to store the coordinates of min and max xy of a label;
	double depthwidthscale[256] = { 0 }; //Stores the values of the width of a block at a certain depth
	double depthheightscale[256] = { 0 }; //Stores the values of the height of a block at a certain depth
	short obj_block_count[EXP_OBJ_NUM] = { 0 };
	double obj_area[EXP_OBJ_NUM] = { 0.0 }; //Stores the approximate area of an object
	Scalar ave_block_depths[EXP_OBJ_NUM];
	double obj_height_min[EXP_OBJ_NUM] = { 10000.0 };
	double obj_width_min[EXP_OBJ_NUM] = { 10000.0 };
	double obj_height_max[EXP_OBJ_NUM] = { 0.0 };
	double obj_width_max[EXP_OBJ_NUM] = { 0.0 };

	distance_block block; //each block (window) that has been scanned; label and coordinates/size is there
	distance_matrix block_matrix; //matrix that contains scanned blocks with their label and other info

	int size_array[LABEL_ARRAYSIZE] = { 0 }; //array that will contain number of windows contained under each label;
											 //position in the array determines the label
											 //by default all zeros
	//Parameters of the sliding window
	int windows_n_rows = BLOCKSIZE;
	int windows_n_cols = BLOCKSIZE;

	// Step of each window; windows will not overlap if stepsize>=blocksize
	int StepSlide = BLOCKSIZE;

	unsigned short CausalMeanArray[5]; //grabs 5 values simultaneously

private:
	//Function to remove all objects which have no blocks assigned to them
	void remove_non_exi_obj()
	{
		for (short i = 0; i <= objcounter; i++)
		{
			if (obj_block_count[i] == 0)
			{
				cout << "Object " << i << " was removed." << endl;
				for (short row = 0; row < 480 / BLOCKSIZE; row++)
				{
					for (short col = 0; col < 640 / BLOCKSIZE; col++)
					{
						short label = block_matrix.get_label_block_row_cols(row, col);
						if (label > i)
						{
							block_matrix.set_label_block_row_cols(row, col, label - 1);
						}
					}
				}
				for (short j = 0; j < EXP_OBJ_NUM; j++)
				{
					short pos = i + j;
					obj_block_count[pos] = obj_block_count[pos + 1];
				}
				--objcounter;
			}
		}
	}

	//function to decide the labelling in grouping
	int logic_process(bool left, bool uppr, bool diag)
	{
		switch (left)
		{
		case true:
		{
			switch (uppr)
			{
			case true:
			{
				switch (diag)
				{
				case true: //all 3 within threshold
				{
					return 7;
					break;
				}
				case false: //left and upper within threshold
				{
					return 6;
					break;
				}
				}
				break;
			}
			case false:
			{
				switch (diag)
				{
				case true://left and daig within threshold
				{
					return 5;
					break;
				}
				case false: //only left within theshold
				{
					return 4;
					break;
				}
				}
				break;
			}
			}
			break;
		}
		case false:
		{
			switch (uppr)
			{
			case true:
			{
				switch (diag)
				{
				case true://upper and diag within threshold
				{
					return 3;
					break;
				}
				case false://only upper within threshold
				{
					return 2;
					break;
				}
				}
				break;
			}
			case false:
			{
				switch (diag)
				{
				case true://only diagonal within theshold
				{
					return 1;
					break;
				}
				case false://none within threshold
				{
					return 0;
					break;
				}
				}
				break;
			}
			}
			break;
		}
		}
	}

	void get_object_block_count()
	{
		for (short row = 0; row < 480 / BLOCKSIZE; row++)
		{
			for (short col = 0; col < 640 / BLOCKSIZE; col++)
			{
				short labelnum = block_matrix.get_label_block_row_cols(row, col);
				obj_block_count[labelnum] += 1;
			}
		}
	}

	//Functions for determining size of blocks (10*10) at certain depths
	double first_order_10(int depth)
	{
		if (depth == 0) { return 0.0; }
		double x = 0.5893*depth - 5.4701;
		return x;
	}

	double second_order_10(int depth)
	{
		if (depth == 0) { return 0.0; }
		double x = -0.0012*pow(depth, 2) + 0.7347*depth - 9.4969;
		return x;
	}

	double third_order_10(int depth)
	{
		if (depth == 0) { return 0.0; }
		double x = -0.0001*pow(depth, 3) + 0.0194*pow(depth, 2) - 0.4638*depth + 11.9323;
		return x;
	}

	//Functions for determining size of blocks (5*5) at certain depths
	double first_order_5(int depth)
	{
		//0.2393   -1.8726
		if (depth == 0) { return 0.0; }
		double x = 0.2393*depth - 1.8726;
		return x;
	}

	double second_order_5(int depth)
	{
		if (depth == 0) { return 0.0; }
		double x = 0.0003*pow(depth, 2) + 0.1846*depth - 0.1115;
		return x;
	}

	double third_order_5(int depth)
	{
		if (depth == 0) { return 0.0; }
		double x = 0.0000*pow(depth, 3) - 0.0010*pow(depth, 2) + 0.2504*depth - 0.9892;
		return x;
	}

	double fourth_order_5(int depth)
	{
		//0.0000   -0.0000    0.0029    0.0601    1.8921
		if (depth == 0) { return 0.0; }
		double x = 0.0000*pow(depth, 4) - 0.0000*pow(depth, 3) + 0.0029*pow(depth, 2) + 0.0601*depth + 1.8921;
		return x;
	}

	void deter_obj_height()
	{
		Mat depth_image = imread("processed_depth_image.png", IMREAD_GRAYSCALE); //stores the image data into opencv matrix object
		for (short i = 0; i <= objcounter; i++)
		{
			if (obj_block_count[i] > 0)
			{
				short rowmin = obj_coords_min[i].y - 1;
				short rowmax = obj_coords_max[i].y + 1;
				
				if (rowmin < 0) { rowmin = 0; }
				if (rowmax > (depth_image.rows / BLOCKSIZE) - 1) { rowmax = depth_image.rows / BLOCKSIZE - 1; }
				
				for (short col = obj_coords_min[i].x; col <= obj_coords_max[i].x; col++)
				{
					double temp_height=0;
					for (short row = rowmin; row <= rowmax; row++)
					{
						int counter = 0;
						double ratio = 1.0;
						Scalar depth = block_matrix.get_average_distance_block_row_cols(row, col);
						if (block_matrix.get_label_block_row_cols(row, col) == i)
						{
							//cout << "Block is of object " << i << endl;
							//conditions to check if the block is at the endge of the object
							bool edge_block = false;
							if (row - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row - 1, col) != i) { edge_block = true; }
							}
							if (row + 1 != depth_image.rows / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row + 1, col) != i) { edge_block = true; }
							}

							//if block is not homogenous and at the edge of the object
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && edge_block == true)
							{
								double max_counter = 0;
								for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
								{
									double stack = 0;
									for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
									{
										int thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { stack++; }
									}
									if (stack > max_counter) { max_counter = stack; }
								}
								ratio = max_counter / StepSlide;
								double dimen = second_order_5(depth[0]);
								temp_height += ratio * dimen;
							}
							else
							{
								double dimen = second_order_5(depth[0]);
								temp_height += dimen;
							}

						}
						else
						{
							bool adjflag = false;

							//is block above that of the object
							if (row - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row - 1, col) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row - 1, col);
								}
							}
							//is block below that of the object
							if (row + 1 != depth_image.rows / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row + 1, col) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row + 1, col);
								}
							}

							//if block is adjacent and not homogenous
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && adjflag == true)
							{
								double max_counter = 0;
								for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
								{
									double stack = 0;
									for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
									{
										int thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { stack++; }
									}
									if (stack > max_counter) { max_counter = stack; }
								}
								ratio = max_counter / StepSlide;
								double dimen = second_order_5(depth[0]);
								temp_height += ratio * dimen;
							}
						}
					}
					if (temp_height > obj_height_max[i]) { obj_height_max[i] = temp_height; }
					if (temp_height < obj_height_min[i]) { obj_height_min[i] = temp_height; }
				}
			}
		}
	}

	void deter_obj_width()
	{
		Mat depth_image = imread("processed_depth_image.png", IMREAD_GRAYSCALE); //stores the image data into opencv matrix object
		for (short i = 0; i <= objcounter; i++)
		{
			if (obj_block_count[i] > 0)
			{
				//cout << "Object " << i << endl;
				short colmin = obj_coords_min[i].x - 1;
				short colmax = obj_coords_max[i].x + 1;

				if (colmin < 0) { colmin = 0; }
				if (colmax > (depth_image.cols / BLOCKSIZE) - 1) { colmax = depth_image.cols / BLOCKSIZE - 1; }

				for (short row = obj_coords_min[i].y; row <= obj_coords_max[i].y; row++)
				{
					//cout << "Entering row " << row << endl;
					double temp_width = 0;
					for (short col = colmin; col <= colmax; col++)
					{
						int counter = 0;
						double ratio = 1.0;
						Scalar depth = block_matrix.get_average_distance_block_row_cols(row, col);
						if (block_matrix.get_label_block_row_cols(row, col) == i)
						{
							//cout << "Block is of object " << i << endl;
							//conditions to check if the block is at the endge of the object
							bool edge_block = false;
							if (col - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row, col - 1) != i) { edge_block = true; }
							}
							if (col + 1 != depth_image.cols / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row, col + 1) != i) { edge_block = true; }
							}

							//if block is not homogenous and at the edge of the object
							if (edge_block == true)
							{
								double max_counter = 0;
								for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
								{
									double stack = 0;
									for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
									{
										short thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { stack++; }
									}
									//cout << "Stack is " << stack << endl;
									if (stack > max_counter) { max_counter = stack; }
								}
								ratio = max_counter / StepSlide;
								double dimen = second_order_5(depth[0]);
								temp_width += ratio * dimen;
								//cout << "depth is " << depth[0] << ", dimen is " << dimen << ", ratio is " << ratio << endl;
								//cout << "temp width now " << temp_width << endl;
							}
							else
							{
								double dimen = second_order_5(depth[0]);
								temp_width += dimen;
								//cout << "depth is " << depth[0] << ", dimen is " << dimen << ", ratio is " << ratio << endl;
								//cout << "temp width now " << temp_width << endl;
							}

						}
						else
						{
							bool adjflag = false;

							//is block left to that of the object
							if (col - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row, col - 1) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row, col - 1);
								}
							}
							//is block right to that of the object
							if (col + 1 != depth_image.cols / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row, col + 1) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row, col + 1);
								}
							}

							//if block is adjacent and not homogenous
							if (adjflag == true)
							{
								double max_counter = 0;
								for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
								{
									double stack = 0;
									for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
									{
										int thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { stack++; }
									}
									if (stack > max_counter) { max_counter = stack; }
								}
								ratio = max_counter / StepSlide;
								double dimen = second_order_5(depth[0]);
								temp_width += ratio * dimen;
								//cout << "depth is " << depth[0] << ", dimen is " << dimen << ", ratio is " << ratio << endl;
								//cout << "temp width now " << temp_width << endl;
							}
						}
					}
					if (temp_width > obj_width_max[i]) { obj_width_max[i] = temp_width; }
					if (temp_width < obj_width_min[i]) { obj_width_min[i] = temp_width; }
				}
			}
		}
	}

	void deter_obj_area_basic()
	{
		for (short obj = 0; obj <= objcounter; obj++)
		{
			for (short col = obj_coords_min[obj].x; col <= obj_coords_max[obj].x; col++)
			{
				for (short row = obj_coords_min[obj].y; row <= obj_coords_max[obj].y; row++)
				{
					if (block_matrix.get_label_block_row_cols(row, col) == obj)
					{
						Scalar depth = block_matrix.get_average_distance_block_row_cols(row, col);
						obj_area[obj] += pow(second_order_5(depth[0]),2);
					}
				}
			}
		}
	}

	void deter_obj_area_inter()
	{
		Mat depth_image = imread("processed_depth_image.png", IMREAD_GRAYSCALE); //stores the image data into opencv matrix object

		for (short obj = 0; obj <= objcounter; obj++)
		{
			for (short col = obj_coords_min[obj].x; col <= obj_coords_max[obj].x; col++)
			{
				for (short row = obj_coords_min[obj].y; row <= obj_coords_max[obj].y; row++)
				{
					if (block_matrix.get_label_block_row_cols(row, col) == obj)
					{
						bool edge_flag = false;
						
						//Is block at the edge of the object
						if (row - 1 != -1)
						{
							if (block_matrix.get_label_block_row_cols(row - 1, col) != obj) { edge_flag = true; }
						}
						if (row + 1 != 480 / StepSlide)
						{
							if (block_matrix.get_label_block_row_cols(row + 1, col) != obj) { edge_flag = true; }
						}
						if (col - 1 != -1)
						{
							if (block_matrix.get_label_block_row_cols(row, col - 1) != obj) { edge_flag = true; }
						}
						if (col + 1 != 640 / StepSlide)
						{
							if (block_matrix.get_label_block_row_cols(row, col + 1) != obj) { edge_flag = true; }
						}

						Scalar depth = block_matrix.get_average_distance_block_row_cols(row, col);
						if (edge_flag)
						{
							short counter = 0;
							for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
							{
								for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
								{
									short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
									if (pixel_depth <= (depth[0] + THRESHOLD) && pixel_depth >= (depth[0] - THRESHOLD)) { counter++; }
								}
							}
							obj_area[obj] += (counter/pow(StepSlide,2))*pow(second_order_5(depth[0]), 2);
						}
						else
						{
							obj_area[obj] += pow(second_order_5(depth[0]), 2);
						}
						
					}
					else
					{
						bool adj_flag = false;
						Scalar depth;
						//Is block adjacent to the object
						if (row - 1 != -1)
						{
							if (block_matrix.get_label_block_row_cols(row - 1, col) == obj)
							{ 
								adj_flag = true;
								depth = block_matrix.get_average_distance_block_row_cols(row - 1, col);
							}
						}
						if (row + 1 != 480 / StepSlide)
						{
							if (block_matrix.get_label_block_row_cols(row + 1, col) == obj)
							{ 
								adj_flag = true;
								depth = block_matrix.get_average_distance_block_row_cols(row + 1, col);
							}
						}
						if (col - 1 != -1)
						{
							if (block_matrix.get_label_block_row_cols(row, col - 1) == obj)
							{
								adj_flag = true; 
								depth = block_matrix.get_average_distance_block_row_cols(row, col - 1);
							}
						}
						if (col + 1 != 640 / StepSlide)
						{
							if (block_matrix.get_label_block_row_cols(row, col + 1) == obj)
							{
								adj_flag = true;
								depth = block_matrix.get_average_distance_block_row_cols(row, col + 1);
							}
						}
						
						if (adj_flag)
						{
							short counter = 0;
							for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
							{
								for (short sub_col = (col*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
								{
									short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
									if (pixel_depth <= (depth[0] + THRESHOLD) && pixel_depth >= (depth[0] - THRESHOLD)) { counter++; }
								}
							}
							obj_area[obj] += (counter / pow(StepSlide, 2))*pow(second_order_5(depth[0]), 2);
						}
					}
				}
			}
		}
	}

	void enforce_boundaries(short low_lim, short up_lim)
	{
		for (short row = 0; row < 480 / StepSlide; row++)
		{
			for (short col = 0; col < 640 / StepSlide; col++)
			{
				Scalar depth = block_matrix.get_average_distance_block_row_cols(row, col);
				if (depth[0] > up_lim || depth[0] < low_lim)
				{
					block_matrix.set_label_block_row_cols(row, col, 0);
					block_matrix.set_average_distance_block_row_cols(row, col, 0);
				}
			}
		}
	}

	void deter_object_depths()
	{
		Scalar temp_obj_depths[EXP_OBJ_NUM] = { 0 };

		for (short row = 0; row < 480 / StepSlide; row++)
		{
			for (short col = 0; col < 640 / StepSlide; col++)
			{
				short label = block_matrix.get_label_block_row_cols(row, col);
				temp_obj_depths[label] += block_matrix.get_average_distance_block_row_cols(row, col);
			}
		}
		for (short obj = 0; obj <= objcounter; obj++)
		{
			ave_block_depths[obj] = temp_obj_depths[obj] / obj_block_count[obj];
		}
	}

public:
	void setup_block_matrix(int width = 640, int height = 480)
	{
		if (!setup) {
			printf("Setting up block matrix...\r\n");
			block_matrix.setup_matrix(width, height, BLOCKSIZE, BLOCKSIZE);
			printf("Done.\r\n");

			printf("Block No. rows = %d, No. cols = %d.\r\n", block_matrix.get_block_rows(), block_matrix.get_block_cols());
			setup = true;
		}
		
	}

	Mat Basic_Segment(Mat LoadedImage){
		
		CausalMeanArray[0] = 0;

		Mat DrawResultGrid = LoadedImage.clone();

		
		
		for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
		{
			for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
			{
						Rect windows(col, row, windows_n_rows, windows_n_cols);

						// Draw grid
						//rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

						// Select windows roi
						Mat Roi = LoadedImage(windows);

						//Here calculate average of each Roi 
						Scalar tempVal = mean(Roi);
						CausalMeanArray[1] = (unsigned short)tempVal.val[0];

						//----- here using header to fill the matrix and blocks --------------	
						Scalar block_avg = tempVal;
						
						int blockrow = (row / BLOCKSIZE);
						int blockcol = (col / BLOCKSIZE);
						
						block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, block_avg);
						short counter = 0;
						int thresh = THRESHOLD;

						//check homogeneity of the block
						for (int i = row; i < row + StepSlide; i++) {
							for (int j = col; j < col + StepSlide; j++) {
								if (DrawResultGrid.at<uchar>(i, j) >= (block_avg[0] + thresh) || DrawResultGrid.at<uchar>(i, j) <= (block_avg[0] - thresh))
								{
									counter++;
								}
							}
						}
						if(counter>0){ block_matrix.set_block_homflag_row_cols(blockrow, blockcol, false); }

						//Changes colour of all pixels to the average
						/*for (int i = row+1; i<=row+StepSlide-1; i++) {
							for (int j = col + 1; j <= col + StepSlide - 1; j++) {
								DrawResultGrid.at<uchar>(i, j) = tempVal[0];
							}
						}*/

			}
		}
		// Save the result from LoadedImage to file
		imwrite("processed_depth_image.png", DrawResultGrid);

			
		
		

		return DrawResultGrid;
	}

	Mat Basic_Colour_Segment(Mat LoadedImage) {

		CausalMeanArray[0] = 0;

		Mat DrawResultGrid = LoadedImage.clone();



		for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
		{
			for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
			{
				Rect windows(col, row, windows_n_rows, windows_n_cols);

				// Draw grid
				//rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

				// Select windows roi
				Mat Roi = LoadedImage(windows);

				//Here calculate average of each Roi 
				cv::Scalar tempVal = mean(Roi);
				CausalMeanArray[1] = (unsigned short)tempVal.val[0];

				//----- here using header to fill the matrix and blocks --------------	
				Scalar block_avg = tempVal;
				

				int blockrow = (row / BLOCKSIZE);
				int blockcol = (col / BLOCKSIZE);

				if (block_avg[0] >= 127) { block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, { 255,255,255 }); }
				else { block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, {0, 0, 0}); }
				
				//Changes colour to either pure white or black
				/*for (int i = row + 1; i <= row + StepSlide - 1; i++) {
					for (int j = col + 1; j <= col + StepSlide - 1; j++) {
						if (block_avg[0] >= 127) { DrawResultGrid.at<uchar>(i, j) = 255; }
						else{ DrawResultGrid.at<uchar>(i, j) = 0; }
					}
				}*/

			}
		}
		// Save the result from LoadedImage to file
		imwrite("processed_colour_image.png", DrawResultGrid);

		return DrawResultGrid;
	}

	void Grouping(Mat LoadedImage, bool colour=false , bool print=false) {
		
		CausalMeanArray[0] = 0;

		Mat DrawResultGrid = LoadedImage.clone();

		printf("Labeling first block...\r\n");
		//st first block label as zero
		block_matrix.set_label_block_row_cols(0, 0, objcounter);

		printf("Labelling first row..\r\n");
		//set labels for all blocks in first row
		for (int col = 1; col < LoadedImage.cols/BLOCKSIZE; col++) {
			
			Scalar curr_block_dist = block_matrix.get_average_distance_block_row_cols(0, col);
			Scalar prev_block_dist = block_matrix.get_average_distance_block_row_cols(0, col - 1);

			if (abs(curr_block_dist[0] - prev_block_dist[0]) <= THRESHOLD) {
				block_matrix.set_label_block_row_cols(0, col, block_matrix.get_label_block_row_cols(0, col - 1));
			}
			else {
				block_matrix.set_label_block_row_cols(0, col, ++objcounter);
			}
		}
		printf("Done.\r\n");
		printf("Labelling first column...\r\n");

		//set labels for all blocks in first column
		for (int row = 1; row < LoadedImage.rows/BLOCKSIZE; row++) {

			Scalar curr_block_dist = block_matrix.get_average_distance_block_row_cols(row, 0);
			Scalar prev_block_dist = block_matrix.get_average_distance_block_row_cols(row, 0);

			if (abs(curr_block_dist[0] - prev_block_dist[0]) <= THRESHOLD) {
				block_matrix.set_label_block_row_cols(row, 0, block_matrix.get_label_block_row_cols(row-1, 0));

				//set min/max coordinates of object
				if (row < obj_coords_min[objcounter].y) { obj_coords_min[objcounter].y = row; }
				if (row > obj_coords_max[objcounter].y) { obj_coords_max[objcounter].y = row; }
			}
			else {
				block_matrix.set_label_block_row_cols(row, 0, ++objcounter);
			}
		}
		printf("Done.\r\n");
		printf("Labelling rest of matrix...\r\n");
		//set labels for rest of the blocks
		for (int row = 1; row < LoadedImage.rows/BLOCKSIZE; row++)
		{
			for (int col = 1; col < LoadedImage.cols/BLOCKSIZE; col++)
			{
				//get average distance values
				Scalar curr_block_dist = block_matrix.get_average_distance_block_row_cols(row, col);
				Scalar left_block_dist = block_matrix.get_average_distance_block_row_cols(row, col-1);
				Scalar uppr_block_dist = block_matrix.get_average_distance_block_row_cols(row-1, col);
				Scalar diag_block_dist = block_matrix.get_average_distance_block_row_cols(row-1, col-1);

				bool thresh_left = (abs(curr_block_dist[0] - left_block_dist[0]) <= THRESHOLD ? true : false);
				bool thresh_uppr = (abs(curr_block_dist[0] - uppr_block_dist[0]) <= THRESHOLD ? true : false);
				bool thresh_diag = (abs(curr_block_dist[0] - diag_block_dist[0]) <= THRESHOLD ? true : false);

				switch (logic_process(thresh_left,thresh_uppr,thresh_diag))
				{
				case 0://none within thresh
				{
					block_matrix.set_label_block_row_cols(row, col, ++objcounter);
					break;
				}
				case 1://Diag within thresh
				{
					int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col - 1);

					block_matrix.set_label_block_row_cols(row, col, trg_obj);
					break;
				}
				case 2://uppr within thresh
				{
					int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col);

					block_matrix.set_label_block_row_cols(row, col, trg_obj);
					break;
				}
				case 3://uppr and diag within thresh
				{
					//uppr closer to current
					if (abs(curr_block_dist[0]-uppr_block_dist[0])<=abs(curr_block_dist[0]-diag_block_dist[0]))
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col - 1, trg_obj);
					}
					//diag closer to current
					else
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col, trg_obj);
					}
					break;
				}
				case 4://left within thresh
				{
					int trg_obj = block_matrix.get_label_block_row_cols(row, col - 1);

					block_matrix.set_label_block_row_cols(row, col, trg_obj);
					break;
				}
				case 5://left and diag within thresh
				{
					//left closer to current
					if (abs(curr_block_dist[0] - left_block_dist[0]) <= abs(curr_block_dist[0] - diag_block_dist[0]))
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col - 1, trg_obj);
					}
					//diag closer to current
					else
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row, col - 1, trg_obj);
					}
					break;
				}
				case 6://left and uppr within thresh
				{
					//uppr closer to current
					if (abs(curr_block_dist[0] - uppr_block_dist[0]) <= abs(curr_block_dist[0] - left_block_dist[0]))
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row, col - 1, trg_obj);
					}
					//left closer to current
					else
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col, trg_obj);
					}
					break;
				}
				case 7://all within thresh
				{
					//uppr closer to current
					if ((abs(curr_block_dist[0] - uppr_block_dist[0]) < abs(curr_block_dist[0] - left_block_dist[0])) && (abs(curr_block_dist[0] - uppr_block_dist[0]) < abs(curr_block_dist[0] - diag_block_dist[0])))
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col - 1, trg_obj);
						block_matrix.set_label_block_row_cols(row, col - 1, trg_obj);
					}
					//diag closer to current
					else if ((abs(curr_block_dist[0] - diag_block_dist[0]) < abs(curr_block_dist[0] - left_block_dist[0])))
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row - 1, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col, trg_obj);
						block_matrix.set_label_block_row_cols(row, col - 1, trg_obj);
					}
					//left closer to curr
					else
					{
						int trg_obj = block_matrix.get_label_block_row_cols(row, col - 1);

						block_matrix.set_label_block_row_cols(row, col, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col - 1, trg_obj);
						block_matrix.set_label_block_row_cols(row - 1, col, trg_obj);
					}
					break;
				}
				default:
					break;
				}
			}
		}	
		printf("Done.\r\n");
		
		//sets all blocks outside this boundary to have the labe zero
		if (!colour) { enforce_boundaries(20, 80); }

		get_object_block_count();

		remove_non_exi_obj();

		if (print)
		{
			printf("Displaying matrix to check filled...\r\n");
			block_matrix.printToScreen();
			printf("Done.\r\n");
		}
		
	}

	void show_min_max_coords()
	{
		for (int i=0; i <= objcounter; i++) 
		{
			if (obj_block_count[i] != 0) {
				cout << "Size of object " << i << " is " << obj_block_count[i] << "." << endl;
				cout << "Min coordinates of object " << i << " is " << obj_coords_min[i] << "." << endl;
				cout << "Max coordinates of object " << i << " is " << obj_coords_max[i] << ".\n" << endl;
			}
		}
	}

	Point* return_max_coords() { return obj_coords_max; }

	Point* return_min_coords() { return obj_coords_min; }

	int return_obj_count() { return objcounter; }

	void add_calib_points(Point* min_coords, Point* max_coords, short object)
	{
		

		cout << "Current Object inside calib points is " << object << "." << endl;
		cout << "Min Coords " << min_coords[object] << "." << endl;
		cout << "Max Coords " << max_coords[object] << "." << endl;

		for (int row = min_coords[object].y; row <= max_coords[object].y; row++)
		{
			for (int col = min_coords[object].x; col <= max_coords[object].x; col++)
			{
				//cout << "Curr Coords (" << col << "," << row << ")." << endl;
				block_matrix.set_label_block_row_cols(row, col, 89+object);

			}
		}
		obj_coords_max[89 + object] = max_coords[object];
		obj_coords_min[89 + object] = min_coords[object];

		cout << "Object_coords_max is now " << obj_coords_max[89 + object] << endl;
		cout << "Object_coords_min is now " << obj_coords_min[89 + object] << endl;

	}

	void print_labels()
	{
		block_matrix.printToScreen();
	}

	void determine_scaling(short num_obj, float object_dimension = 145, bool print_scale = false)
	{
		
		cout << "Determining Scaling" << endl;
		int dif_dis_arr[2];
		
		Point center_block;
		center_block.x = (obj_coords_min[89 + num_obj].x + (obj_coords_max[89 + num_obj].x - obj_coords_min[89 + num_obj].x) / 2);
		center_block.y = (obj_coords_min[89 + num_obj].y + (obj_coords_max[89 + num_obj].y - obj_coords_min[89 + num_obj].y) / 2);

		Scalar center_depth_block = block_matrix.get_average_distance_block_row_cols(center_block.y, center_block.x);

		cout << "Object " << num_obj << " max coordinates are " << obj_coords_max[89 + num_obj] << "." << endl;
		cout << "Object " << num_obj << " min coordinates are " << obj_coords_min[89 + num_obj] << "." << endl;
		cout << "Object dimension " << object_dimension << ", width difference " << obj_coords_max[89 + num_obj].x - obj_coords_min[89 + num_obj].x << endl;
		cout << "Object dimension " << object_dimension << ", height difference " << obj_coords_max[89 + num_obj].y - obj_coords_min[89 + num_obj].y << endl;

		float width = object_dimension / (obj_coords_max[89 + num_obj].x - obj_coords_min[89 + num_obj].x);
		float height = object_dimension / (obj_coords_max[89 + num_obj].y - obj_coords_min[89 + num_obj].y);

		dif_dis_arr[num_obj - 1] = center_depth_block[0];
		depthwidthscale[dif_dis_arr[num_obj - 1]] = width;
		depthheightscale[dif_dis_arr[num_obj - 1]] = height;

		cout << "Object " << num_obj << ": Depth " << center_depth_block[0] << ", Block Height " << height << "mm, Block Width " << width << "mm." << endl;
			
		short check = 0;
		cout << "Do you want to calibrate to file?: \r\n1 Yes\r\n2 No\r\n";
		cin >> check;

		if (check == 1)
		{
			cout << "Writing to scale_data.csv." << endl;
			fout.open("scale_data52.csv", ios::app);
			fout << center_depth_block[0] << "," << height << "," << width << endl;
			fout.close();
		}
			

		if (print_scale)
		{
			for (int i = 0; i < 256; i++)
			{
				cout << "Scale for width at depth " << i << " is " << depthwidthscale[i] << endl;
				cout << "Scale for height at depth " << i << " is " << depthheightscale[i] << endl;
			}
		}
	}

	//not used and doesnt work very well
	void reset_matrix()
	{
		block_matrix.reset_blocks();
		objcounter = 0; //by default 1 object (there will always be something)
		obj_coords_max[250] = { 0,0 };
		obj_coords_min[250] = { 0,0 }; //array to store the coordinates of min and max xy of a label;
		depthwidthscale[256] = { 0 };	//Stores the values of the width of a block at a certain depth
		depthheightscale[256] = { 0 };	//Stores the values of the height of a block at a certain depth

		size_array[LABEL_ARRAYSIZE] = { 0 };	//array that will contain number of windows contained under each label;
												//position in the array determines the label
												//by default all zeros
	}

	void set_approx_obj_coords()
	{
		for (short i = 0; i < EXP_OBJ_NUM; i++) //sets min to be high as default value of array is [0,0]
		{
			obj_coords_min[i].x = 1000;
			obj_coords_min[i].y = 1000;

		}
		for (short row = 0; row < 480 / BLOCKSIZE; row++)
		{
			for (short col = 0; col < 640 / BLOCKSIZE; col++)
			{
				short curr_obj = block_matrix.get_label_block_row_cols(row, col);
				if (col < obj_coords_min[curr_obj].x) { obj_coords_min[curr_obj].x = col; }
				if (row < obj_coords_min[curr_obj].y) { obj_coords_min[curr_obj].y = row; }
				if (col > obj_coords_max[curr_obj].x) { obj_coords_max[curr_obj].x = col; }
				if (row > obj_coords_max[curr_obj].y) { obj_coords_max[curr_obj].y = row; }
			}
		}
	}

	void deter_obj_dimen()
	{
		//deter_obj_area_basic();
		deter_obj_area_inter();
		deter_object_depths();
		deter_obj_height();
		deter_obj_width();
	}

	void print_obj_areas()
	{
		for (short i = 0; i <= objcounter; i++)
		{
			if (obj_block_count[i] > 4)
			{ 
				cout << "Object " << i << " has " << obj_block_count[i] << " blocks, " << obj_coords_min[i] << ", " << obj_coords_max[i] << "." << endl;
				cout << "Average depth of " << ave_block_depths[i][0];
				cout  << " and Approximate area of " << obj_area[i] << "mm^2 or " << obj_area[i] / 1000000 << "m^2." << endl;
				cout << "Max height " << obj_height_max[i] << "mm, min height " << obj_height_min[i] << "mm.";
				cout << " Max width " << obj_width_max[i] << "mm, min width " << obj_width_min[i] << "mm.\n" << endl;
			}
		}
	}

	double* return_obj_heights() { return obj_height_max; }
	
	double* return_obj_width() { return obj_width_max; }

	Scalar* return_obj_depths() { return ave_block_depths; }

	};