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
	double obj_height_min[EXP_OBJ_NUM] = { 10000000.0 };
	double obj_width_min[EXP_OBJ_NUM] = { 10000000.0 };
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
		double x = 0.0010*pow(depth, 2) + 0.1080*depth + 1.9215;
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

	void deter_obj_area()
	{
		Mat depth_image = imread("processed_depth_image.png", IMREAD_GRAYSCALE); //stores the image data into opencv matrix object
		//cout << "Number of Objects " << objcounter << endl;
		//cout << "Determing object areas..." << endl;
		for (short i = 0; i <= objcounter; i++)
		{
			//cout << "Object " << i << " has " << obj_block_count[i] << " blocks and coords " << obj_coords_min[i] << obj_coords_max[i] << endl;

			if (obj_block_count[i] > 0)
			{
				//cout << "Aprroximating Area..." << endl;
				short rowmin = obj_coords_min[i].y - 1;
				short rowmax = obj_coords_max[i].y + 1;
				short colmin = obj_coords_min[i].x - 1;
				short colmax = obj_coords_max[i].x + 1;

				if (rowmin < 0) { rowmin = 0; }
				if (rowmax > (depth_image.rows / BLOCKSIZE)) { rowmax = depth_image.rows / BLOCKSIZE; }
				if (colmin < 0) { colmin = 0; }
				if (colmax > (depth_image.cols / BLOCKSIZE)) { colmax = depth_image.cols / BLOCKSIZE; }

				for (short row = rowmin; row < rowmax; row++)
				{
					for (short col = colmin; col < colmax; col++)
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
							if (col - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row, col - 1) != i) { edge_block = true; }
							}
							if (col + 1 != depth_image.cols / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row, col + 1) != i) { edge_block = true; }
							}

							//if block is not homogenous and at the edge of the object
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && edge_block == true)
							{
								//cout << "Closer inspection required of block " << row << "," << col << endl;
								for (short sub_row = (row * StepSlide); sub_row < (row * StepSlide + StepSlide); sub_row++)
								{
									for (short sub_col = (col * StepSlide); sub_col < (col * StepSlide + StepSlide); sub_col++)
									{
										//cout << i << "," << col << "." << sub_col << "," << row << "." << sub_row << endl;
										int thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { counter++; }
									}
								}
								ratio = counter / pow(StepSlide, 2);
								double dimen = second_order_5(depth[0]);
								obj_area[i] += ratio * dimen * dimen;
							}
							else
							{
								double dimen = second_order_5(depth[0]);
								obj_area[i] += dimen * dimen;
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
							//is block to the left of the object
							if (col - 1 != -1)
							{
								if (block_matrix.get_label_block_row_cols(row, col - 1) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row, col - 1);
								}
							}
							//is block to the right of the object
							if (col + 1 != depth_image.cols / BLOCKSIZE)
							{
								if (block_matrix.get_label_block_row_cols(row, col + 1) == i)
								{
									adjflag = true;
									depth = block_matrix.get_average_distance_block_row_cols(row, col + 1);
								}
							}

							//if block is adjacent and not homogenous
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && adjflag == true)
							{
								//cout << "Block is not of object " << i << ", but is adjacent." << endl;
								for (short sub_row = (row * StepSlide); sub_row < (row * StepSlide + StepSlide); sub_row++)
								{
									for (short sub_col = (col * StepSlide); sub_col < (col * StepSlide + StepSlide); sub_col++)
									{
										//cout << i << "," << col << "." << sub_col << "," << row << "." << sub_row << endl;
										int thresh = THRESHOLD;
										short pixel_depth = depth_image.at<uchar>(sub_row, sub_col);
										if (pixel_depth <= (depth[0] + thresh) && pixel_depth >= (depth[0] - thresh)) { counter++; }
									}
								}
								ratio = counter / pow(StepSlide, 2);
								double dimen = second_order_5(depth[0]);
								obj_area[i] += ratio * dimen * dimen;
							}
						}
					}
				}
				cout << "Done." << endl;
			}
		}
		cout << "Done." << endl;
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
				if (rowmax > (depth_image.rows / BLOCKSIZE)) { rowmax = depth_image.rows / BLOCKSIZE; }
				
				for (short col = obj_coords_min[i].x; col < obj_coords_max[i].x; col++)
				{
					double temp_height=0;
					for (short row = rowmin; row < rowmax; row++)
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
				short colmin = obj_coords_min[i].x - 1;
				short colmax = obj_coords_max[i].x + 1;

				if (colmin < 0) { colmin = 0; }
				if (colmax > (depth_image.cols / BLOCKSIZE)) { colmax = depth_image.cols / BLOCKSIZE; }

				for (short row = obj_coords_min[i].y; row < obj_coords_max[i].y; row++)
				{
					double temp_width = 0;
					for (short col = colmin; col < colmax; col++)
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
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && edge_block == true)
							{
								double max_counter = 0;
								for (short sub_row = (row*StepSlide); sub_row < (row*StepSlide + StepSlide); sub_row++)
								{
									double stack = 0;
									for (short sub_col = (row*StepSlide); sub_col < (col*StepSlide + StepSlide); sub_col++)
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
							}
							else
							{
								double dimen = second_order_5(depth[0]);
								temp_width += dimen;
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
							if (block_matrix.get_block_homflag_row_cols(row, col) == false && adjflag == true)
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
							obj_area[obj] += (counter/pow(StepSlide,2))*pow(first_order_5(depth[0]), 2);
						}
						else
						{
							obj_area[obj] += pow(first_order_5(depth[0]), 2);
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
							obj_area[obj] += (counter / pow(StepSlide, 2))*pow(first_order_5(depth[0]), 2);
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

public:
	//prev students code modified 
	//OUTDATED
	Mat Segmenting(Mat LoadedImage)
	{
		namedWindow("1. Loaded img", WINDOW_AUTOSIZE);
		imshow("1. Loaded img", LoadedImage);
		waitKey(1);

		// Save the result from LoadedImage to file
		imwrite("Step1_boxes.png", LoadedImage);



		CausalMeanArray[0] = 0;

		Mat DrawResultGrid = LoadedImage.clone();

		//block_matrix.setup_matrix(LoadedImage.rows, LoadedImage.cols, windows_n_cols, windows_n_rows); 
		//block_matrix.setup_matrix_defaultvals();
		block_matrix.setup_matrix(640, 480, BLOCKSIZE, BLOCKSIZE);
		block_matrix.printToScreen();
		std::cout << endl;


		/*std::cout << "before proceding. print block rows: " << block_matrix.get_block_rows() << endl;
		std::cout << "before proceding. print block columns: " << block_matrix.get_block_cols() << endl;
		std::cout << endl;*/

#ifdef option2

		for (int row = 0; row <= LoadedImage.rows - windows_n_rows; row += StepSlide)
		{

			for (int col = 0; col <= LoadedImage.cols - windows_n_cols; col += StepSlide)
			{


				//if one of the first blocks (see description in thesis pdf)
				if (row < BLOCKSIZE || col < BLOCKSIZE) {


					//very first block. Scan only current block
					if (row < BLOCKSIZE && col < BLOCKSIZE) {


						Rect windows(col, row, windows_n_rows, windows_n_cols);
						Mat DrawResultHere = LoadedImage.clone();

						// Draw only rectangle
						rectangle(DrawResultHere, windows, Scalar(255, 0, 255), 1, 8, 0);
						// Draw grid
						rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);
						// Select windows roi
						Mat Roi = LoadedImage(windows);

						//function to replace previous print, updates windows
						update_image(DrawResultHere, DrawResultGrid);


						//Here calculate average of each Roi 
						cv::Scalar tempVal = mean(Roi);
						//cv::Scalar myMAtMean = tempVal;
						CausalMeanArray[1] = (unsigned short)tempVal.val[0];
						std::ostringstream str1;
						str1 << CausalMeanArray[1];

						//----- here using header to fill the matrix and blocks --------------	
						Scalar block_avg = tempVal;
						ostringstream str1_label; //for ROI


						int rowmultiplier = row / BLOCKSIZE;// + 1;
						int colmultiplier = col / BLOCKSIZE;// + 1;

						int blockrow = (row / BLOCKSIZE);// + 1;
						int blockcol = (col / BLOCKSIZE);// + 1;

						block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, block_avg);

#ifdef PrintMode
						cout << "Simple extraction: " << tempVal << endl;
						cout << "Row: " << row << ",    " << "Column: " << col << endl;
						cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						cout << endl;
#endif

						objcounter++;
						//-------- using the matrix header
						block_matrix.set_label_block_row_cols(blockrow, blockcol, objcounter);
						//create string with the label to print 

						str1_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
						//print label into the image
						cv::putText(Roi, str1_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);//????

						//----------------------------------

						//print the average value into the image
						//after all the cases, print the label of the cell and its average value
						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = 1;
					}



					//any block in the first row. Scan only left block + current block
					if (row < BLOCKSIZE && col >= BLOCKSIZE) {

						Rect windows(col, row, windows_n_rows, windows_n_cols);
						Rect windowLeft(col - BLOCKSIZE, row, windows_n_rows, windows_n_cols);

						Mat DrawResultHere = LoadedImage.clone();

						// Draw only rectangle
						rectangle(DrawResultHere, windows, Scalar(255, 0, 255), 1, 8, 0);//  magenta/fuchsia (here-black)
						rectangle(DrawResultHere, windowLeft, Scalar(255), 1, 8, 0);

						rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);//  magenta/fuchsia (here-black)	

																				 // Select windows roi
						Mat Roi = LoadedImage(windows);
						Mat RoiLeft = LoadedImage(windowLeft);


						//function to replace print, updates windows
						update_image(DrawResultHere, DrawResultGrid);



						//Here calculate average of each Roi 
						//1) CURRENT ROI
						cv::Scalar tempVal = mean(Roi);
						CausalMeanArray[1] = (unsigned short)tempVal.val[0];
						std::ostringstream str1;
						str1 << CausalMeanArray[1];
						Scalar block_avg = tempVal;
						ostringstream str1_label; //for ROI


						int rowmultiplier = row / BLOCKSIZE;// + 1;
						int colmultiplier = col / BLOCKSIZE;// + 1;
															//int blockrow = rowmultiplier + (block_matrix.get_block_cols() * rowmultiplier) + 1; 
						int blockrow = (row / BLOCKSIZE);// + 1;
						int blockcol = (col / BLOCKSIZE);// + 1;
														 //int blockcol = (col / 60)+1;
						block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, block_avg);

#ifdef PrintMode
						cout << "Simple extraction: " << tempVal << endl;
						cout << "Row: " << row << ",    " << "Column: " << col << endl;
						cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						cout << endl;
#endif
						//---------------------------------------------------------------------

						//ROI OF THE LEFT BLOCK
						cv::Scalar tempValLeft = mean(RoiLeft);
						CausalMeanArray[0] = (unsigned short)tempValLeft.val[0];
						std::ostringstream str0;
						str0 << CausalMeanArray[0];
						//----- here using header to fill the matrix and blocks --------------			
						Scalar block_avg_left = tempValLeft;
						int blockrowLeft = blockrow;
						int blockcolLeft = blockcol - 1;
						block_matrix.set_average_distance_block_row_cols(blockrowLeft, blockcolLeft, block_avg_left);

#ifdef PrintMode
						cout << "Simple extraction. LEFT: " << tempValLeft << endl;
						cout << "Row: " << row << ",    " << "Column: " << col - 1 << endl;
						cout << "BlockRow: " << blockrowLeft << ",    " << "BlockColumn: " << blockcolLeft << endl;
						//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
						cout << endl;
#endif


						unsigned short diff_left;
						diff_left = abs(CausalMeanArray[1] - CausalMeanArray[0]);

						//if different from before, print in random colour + update label and object counter
						if (diff_left > THRESHOLD) {

							objcounter++;
							//-------- using the matrix header
							block_matrix.set_label_block_row_cols(blockrow, blockcol, objcounter);
							//create string with the label to print 

							str1_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							cv::putText(Roi, str1_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = 1;
						}

						else if (diff_left <= THRESHOLD) {

							short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

							ostringstream str_LEFT_label, str_LEFT_detect;
							str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							//				str_LEFT_detect << "L"; //indicates that it's similar to the one on the left
							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//				cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;
						} //end else if


					} //end if statement when block R1 any column





					  //any row first column block. Scan only upper block + current block
					if (row >= BLOCKSIZE && col < BLOCKSIZE) {

						Rect windows(col, row, windows_n_rows, windows_n_cols);
						Rect windowUp(col, row - BLOCKSIZE, windows_n_rows, windows_n_cols);
						Mat DrawResultHere = LoadedImage.clone();


						// Draw only rectangle
						rectangle(DrawResultHere, windows, Scalar(255, 0, 255), 1, 8, 0);//  magenta/fuchsia (here-black)
						rectangle(DrawResultHere, windowUp, Scalar(255), 1, 8, 0);

						// Draw grid
						rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

						// Select windows roi
						Mat Roi = LoadedImage(windows);
						Mat RoiUp = LoadedImage(windowUp);

						//function to replace previous print, updates windows
						update_image(DrawResultHere, DrawResultGrid);

						//Here calculate average of each Roi /-Olya
						cv::Scalar tempVal = mean(Roi);
						//cv::Scalar myMAtMean = tempVal;
						CausalMeanArray[1] = (unsigned short)tempVal.val[0];
						std::ostringstream str1;
						str1 << CausalMeanArray[1];
						//----- here using header to fill the matrix and blocks --------------	
						Scalar block_avg = tempVal;
						ostringstream str1_label; //for ROI


						int rowmultiplier = row / BLOCKSIZE;// +1;
						int colmultiplier = col / BLOCKSIZE;// + 1;

						int blockrow = (row / BLOCKSIZE);// + 1;
						int blockcol = (col / BLOCKSIZE);// + 1;

						block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, block_avg);

#ifdef PrintMode
						cout << "Simple extraction: " << tempVal << endl;
						cout << "Row: " << row << ",    " << "Column: " << col << endl;
						cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						cout << endl;
#endif


						cv::Scalar tempValUp = mean(RoiUp);
						CausalMeanArray[3] = (unsigned short)tempValUp.val[0];
						std::ostringstream str3;
						str3 << CausalMeanArray[3];

						//----- here using header to fill the matrix and blocks --------------			
						Scalar block_avg_up = tempValUp;
						int blockrowUp = blockrow - 1;
						int blockcolUp = blockcol;
						block_matrix.set_average_distance_block_row_cols(blockrowUp, blockcolUp, block_avg_up);

#ifdef PrintMode
						cout << "Simple extraction. UP: " << tempValUp << endl;
						cout << "Row: " << row - 1 << ",    " << "Column: " << col << endl;
						cout << "BlockRow: " << blockrowUp << ",    " << "BlockColumn: " << blockcolUp << endl;
						//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
						cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrowUp, blockcolUp) << endl;
						cout << endl;
#endif



						//mean differences between windows
						unsigned short diff_up;
						diff_up = abs(CausalMeanArray[1] - CausalMeanArray[3]);

						//new object -> new label
						if (diff_up > THRESHOLD) {
							//update object counter from the previous iteration
							objcounter++;

							//--------using the matrix header
							block_matrix.set_label_block_row_cols(blockrow, blockcol, objcounter);
							//create string with the label to print 
							str1_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							cv::putText(Roi, str1_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = 1;

						} //end if difference between current and upper blocks bigger than threshold



						  //similar average distance of the current and the upper blocks
						if (diff_up <= THRESHOLD) {

							short getLabelUP = block_matrix.get_label_block_row_cols(blockrowUp, blockcolUp);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUP);

							ostringstream str_UP_label, str_UP_detect;
							str_UP_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							cv::putText(Roi, str_UP_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}//end if difference between current and upper blocks is smaller than treshold


					} //enf if R1 C1 block

				} //end of the unusual cases





				  //if normal position
				else if (row >= BLOCKSIZE && col >= BLOCKSIZE) {
					// resulting window
					Rect windows(col, row, windows_n_rows, windows_n_cols);
					//My added causal  scanning windows
					Rect windowUpLeft(col - BLOCKSIZE, row - BLOCKSIZE, windows_n_rows, windows_n_cols);
					Rect windowUp(col, row - BLOCKSIZE, windows_n_rows, windows_n_cols);
					Rect windowLeft(col - BLOCKSIZE, row, windows_n_rows, windows_n_cols);

					Mat DrawResultHere = LoadedImage.clone();

					// Draw only rectangle
					rectangle(DrawResultHere, windows, Scalar(255, 0, 255), 1, 8, 0);//  magenta/fuchsia (here-black)																	 // My added windows to scan
					rectangle(DrawResultHere, windowUpLeft, Scalar(255), 1, 8, 0);
					rectangle(DrawResultHere, windowUp, Scalar(255), 1, 8, 0);
					rectangle(DrawResultHere, windowLeft, Scalar(255), 1, 8, 0);

					// Draw grid
					rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

					// Select windows roi
					Mat Roi = LoadedImage(windows);
					Mat RoiUpLeft = LoadedImage(windowUpLeft);
					Mat RoiUp = LoadedImage(windowUp);
					Mat RoiLeft = LoadedImage(windowLeft);

					//function to replace old print to update window
					update_image(DrawResultHere, DrawResultGrid);


					cv::Scalar tempVal = mean(Roi);
					CausalMeanArray[1] = (unsigned short)tempVal.val[0];
					std::ostringstream str1;
					str1 << CausalMeanArray[1];
					//----- here using header to fill the matrix and blocks --------------
					Scalar block_avg = tempVal;
					ostringstream str1_label; //for ROI


					int rowmultiplier = row / BLOCKSIZE;// + 1;
					int colmultiplier = col / BLOCKSIZE;// + 1;
					int blockrow = (row / BLOCKSIZE);// + 1;
					int blockcol = (col / BLOCKSIZE);// + 1;
					block_matrix.set_average_distance_block_row_cols(blockrow, blockcol, block_avg);

#ifdef PrintMode
					cout << "Simple extraction: " << tempVal << endl;
					cout << "Row: " << row << ",    " << "Column: " << col << endl;
					cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
					//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
					cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
					cout << endl;
#endif

					//2) Left window
					cv::Scalar tempValLeft = mean(RoiLeft);
					CausalMeanArray[0] = (unsigned short)tempValLeft.val[0];
					std::ostringstream str0;
					str0 << CausalMeanArray[0];
					//----- here using header to fill the matrix and blocks --------------
					Scalar block_avg_left = tempValLeft;
					int blockrowLeft = blockrow;
					int blockcolLeft = blockcol - 1;
					block_matrix.set_average_distance_block_row_cols(blockrowLeft, blockcolLeft, block_avg_left);

#ifdef PrintMode
					cout << "Simple extraction. LEFT: " << tempValLeft << endl;
					cout << "Row: " << row << ",    " << "Column: " << col - 1 << endl;
					cout << "BlockRow: " << blockrowLeft << ",    " << "BlockColumn: " << blockcolLeft << endl;
					//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
					cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
					cout << endl;
#endif


					//3)Upleft window
					cv::Scalar tempValUpLeft = mean(RoiUpLeft);
					CausalMeanArray[2] = (unsigned short)tempValUpLeft.val[0];
					std::ostringstream str2;
					str2 << CausalMeanArray[2];

					//----- here using header to fill the matrix and blocks --------------
					Scalar block_avg_upleft = tempValUpLeft;
					int blockrowUpLeft = blockrow - 1;
					int blockcolUpLeft = blockcol - 1;
					block_matrix.set_average_distance_block_row_cols(blockrowUpLeft, blockcolUpLeft, block_avg_upleft);

#ifdef PrintMode
					cout << "Simple extraction. UPLEFT: " << tempValUpLeft << endl;
					cout << "Row: " << row - 1 << ",    " << "Column: " << col - 1 << endl;
					cout << "BlockRow: " << blockrowUpLeft << ",    " << "BlockColumn: " << blockcolUpLeft << endl;
					//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
					cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrowUpLeft, blockcolUpLeft) << endl;
					cout << endl;
#endif


					//4)Upper window
					cv::Scalar tempValUp = mean(RoiUp);
					CausalMeanArray[3] = (unsigned short)tempValUp.val[0];
					std::ostringstream str3;
					str3 << CausalMeanArray[3];

					//----- here using header to fill the matrix and blocks --------------
					Scalar block_avg_up = tempValUp;
					int blockrowUp = blockrow - 1;
					int blockcolUp = blockcol;
					block_matrix.set_average_distance_block_row_cols(blockrowUp, blockcolUp, block_avg_up);

#ifdef PrintMode
					cout << "Simple extraction. UP: " << tempValUp << endl;
					cout << "Row: " << row - 1 << ",    " << "Column: " << col << endl;
					cout << "BlockRow: " << blockrowUp << ",    " << "BlockColumn: " << blockcolUp << endl;
					//cout << "BlockRow: " << blockrow << ",    " << "BlockColumn: " << blockcol << endl;
					cout << "Average value assigned to this block: " << block_matrix.get_average_distance_block_row_cols(blockrowUp, blockcolUp) << endl;
					cout << endl;
#endif
					// assigning average values finishes here


					//mean differences between windows 

					unsigned short diff_left;
					diff_left = abs(CausalMeanArray[1] - CausalMeanArray[0]);

					unsigned short diff_upleft;
					diff_upleft = abs(CausalMeanArray[1] - CausalMeanArray[2]);

					unsigned short diff_up;
					diff_up = abs(CausalMeanArray[1] - CausalMeanArray[3]);



					//if all 3 are different
					if (((diff_left > THRESHOLD) && (diff_up > THRESHOLD) && (diff_upleft > THRESHOLD))) {
						//update object counter from previous iteration
						objcounter++;

						//-------- using the matrix header
						block_matrix.set_label_block_row_cols(blockrow, blockcol, objcounter);
						//create string with the label to print

						str1_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
						cv::putText(Roi, str1_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = 1;

					} //end if all 4 different





					  //if all four similar values
					else if ((diff_left <= THRESHOLD) && (diff_upleft <= THRESHOLD) && (diff_up <= THRESHOLD)) {


						// collect labels of all surrounding blocks
						//short label_array[4];
						short first = block_matrix.get_label_block_row_cols(blockrowUpLeft, blockcolUpLeft); //upleft = array position 1
						short second = block_matrix.get_label_block_row_cols(blockrowUp, blockcolUp);  //up = array position 2
						short third = block_matrix.get_label_block_row_cols(blockcolLeft, blockcolLeft);   //left = array position 3


																										   //if all three with the same label; take left by default
						if (first == second == third) {


							//blockrowUpLeft
							short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

							ostringstream str_LEFT_label, str_LEFT_detect;
							str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}


						//if labels of upper and upper-left similar; upleft taken
						else if (first = second) {

							short getLabelUP_UPLEFT = block_matrix.get_label_block_row_cols(blockrowUpLeft, blockcolUpLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUP_UPLEFT);

							ostringstream str_UP_UPLEFT_label, str_UP_UPLEFT_detect;
							str_UP_UPLEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							//			str_UP_UPLEFT_detect << "UUL";
							cv::putText(Roi, str_UP_UPLEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}


						//left and upleft; left taken
						else if (first = third) {

							short getLabelUPLEFT_LEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUPLEFT_LEFT);

							ostringstream str_UPLEFT_LEFT_label, str_UPLEFT_LEFT_detect;
							str_UPLEFT_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

							cv::putText(Roi, str_UPLEFT_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}


						//up and left; left taken
						else if (second == third) {

							short getLabelUP_LEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUP_LEFT);

							ostringstream str_UP_LEFT_label, str_UP_LEFT_detect;
							str_UP_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							//			str_UP_LEFT_detect << "U-L";
							cv::putText(Roi, str_UP_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//			cv::putText(Roi, str_UP_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}

						//when all three different labels-> prefer left label
						else if (first != second != third) {

							short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
							block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

							ostringstream str_LEFT_label, str_LEFT_detect;
							str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
							//			str_LEFT_detect << "all";
							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//			cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);

							cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
							size_array[current_label] = size_array[current_label] + 1;

						}


					}//if all four have similar values


					 //when both left and upleft are similar -> prefer left
					else if ((diff_left <= THRESHOLD) && (diff_upleft <= THRESHOLD)) {
						short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

						ostringstream str_LEFT_label, str_LEFT_detect;
						str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}


					//when both upleft and up similar -> prefer up
					else if ((diff_upleft <= THRESHOLD) && (diff_up <= THRESHOLD)) {
						short getLabelUp = block_matrix.get_label_block_row_cols(blockrowUp, blockcolUp);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUp);

						ostringstream str_Up_label, str_Up_detect;
						str_Up_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
						//				str_Up_detect << "ulu";
						cv::putText(Roi, str_Up_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
						//				cv::putText(Roi, str_Up_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}

					//when both left and up similar -> prefer left
					else if ((diff_left <= THRESHOLD) && (diff_up <= THRESHOLD)) {

						short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

						ostringstream str_LEFT_label, str_LEFT_detect;
						str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);
						//					str_LEFT_detect << "lul";
						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
						//					cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}

					//just left similar
					else if (diff_left <= THRESHOLD) {
						short getLabelLEFT = block_matrix.get_label_block_row_cols(blockrowLeft, blockcolLeft);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelLEFT);

						ostringstream str_LEFT_label, str_LEFT_detect;
						str_LEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}

					//just upleft similar
					else if (diff_upleft <= THRESHOLD) {
						short getLabelUPLEFT = block_matrix.get_label_block_row_cols(blockrowUpLeft, blockcolUpLeft);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUPLEFT);

						ostringstream str_UPLEFT_label, str_UPLEFT_detect;
						str_UPLEFT_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

						cv::putText(Roi, str_UPLEFT_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}

					//just up similar
					else if (diff_up <= THRESHOLD) {
						short getLabelUP = block_matrix.get_label_block_row_cols(blockrowUp, blockcolUp);
						block_matrix.set_label_block_row_cols(blockrow, blockcol, getLabelUP);

						ostringstream str_UP_label, str_UP_detect;
						str_UP_label << block_matrix.get_label_block_row_cols(blockrow, blockcol);

						cv::putText(Roi, str_UP_label.str(), cv::Point(7, 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						cv::putText(Roi, str1.str(), cv::Point(7, 7), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						int current_label = block_matrix.get_label_block_row_cols(blockrow, blockcol);
						size_array[current_label] = size_array[current_label] + 1;

					}


					//something not covered


					else {
						//std::cout << "NONE OF THE ABOVE" << endl;
					}


				} //end ifrow >=60 and col >=60



			}//end columns


		}//end rows




		for (int i = 0; i < LABEL_ARRAYSIZE; i++) {
			if (size_array[i] == 0)
				std::cout << "NAN" << "-";

			else
				std::cout << size_array[i] << "-";
		}
		std::cout << endl;

		return DrawResultGrid;

#endif
	}//end segmenting()
	//OUTDATEd
	void update_image(Mat DrawResultHere, Mat DrawResultGrid) {
		namedWindow("2. Draw Rectangle", WINDOW_AUTOSIZE);
		imshow("2. Draw Rectangle", DrawResultHere);
		waitKey(1);
		imwrite("Step2_boxes.png", DrawResultHere);

		// Show grid
		namedWindow("3. Show Grid", WINDOW_AUTOSIZE);
		imshow("3. Show Grid", DrawResultGrid);
		waitKey(1);
		imwrite("Step3_boxes.png", DrawResultGrid);
	}

	void setup_block_matrix(int width = 640, int height = 480, int blockwidth = BLOCKSIZE, int blockheight = BLOCKSIZE)
	{
		if (!setup) {
			printf("Setting up block matrix...\r\n");
			block_matrix.setup_matrix(width, height, blockwidth, blockheight);
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
				rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

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

	void Grouping(Mat LoadedImage, bool print=false) {
		
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
		enforce_boundaries(20,120);

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
		//deter_obj_area();
		//deter_obj_height();
		//deter_obj_width();
	}


	void print_obj_areas()
	{
		for (short i = 0; i <= objcounter; i++)
		{
			if (obj_block_count[i] != 0)
			{ 
				cout << "Object " << i << " has " << obj_block_count[i] << " blocks, " << obj_coords_min[i] << ", " << obj_coords_max[i] << "." << endl;
				cout  << "Approximate area of " << obj_area[i] << "mm^2 or " << obj_area[i] / 1000000 << "m^2." << endl;
				cout << "Max height " << obj_height_max[i] << "mm, min height " << obj_height_min[i] << "mm.";
				cout << " Max width " << obj_width_max[i] << "mm, min width " << obj_width_min[i] << "mm.\n" << endl;
			}
		}
	}

	

	};