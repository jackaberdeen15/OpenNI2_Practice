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
#define BLOCKSIZE 40
#define THRESHOLD 7
#define LABEL_ARRAYSIZE 150 //typically depends on how many labels to be stored (depends on the scanning window size)
using namespace cv;
using namespace std;

class Image_Process {
protected:
	//  Load the image from file
	Mat LoadedImage;
	//LoadedImage = imread("GRABBED.png", IMREAD_COLOR);
	short objcounter = 0; //by default 1 object (there will always be something)

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

public:
	void Segmenting(Mat LoadedImage)
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
						cv::putText(Roi, str1_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);//????

						//----------------------------------
						//block_matrix.printToScreen();
						//cout << endl;


						/*std::cout << "FIRST BLOCK. New object at pixels: " << windows.y << " , " << windows.x << endl;
						std::cout << "Object Number: " << objcounter << endl;
						std::cout << "Average distance of this object: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/

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
							cv::putText(Roi, str1_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);



							//objsquares = 0;  //FOR LATER, WHEN MORE COMPLEX
							/*std::cout << "R1 block. New object at pixels: " << windows.y << " , " << windows.x << endl;
							std::cout << "Object Number: " << objcounter << endl;
							std::cout << "Average distance of this object: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/

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
							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//				cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);


							//print new label of the object
							/*std::cout << "One of the first blocks. Block is assigned left object's label: " << getLabelLEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/

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
							cv::putText(Roi, str1_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


							/*std::cout << "R2 C1. New object at pixels: " << windows.y << " , " << windows.x << endl;
							std::cout << "Object Number: " << objcounter << endl;
							std::cout << "Average distance of this object: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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
							cv::putText(Roi, str_UP_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


							//print new label of the object
							/*std::cout << "Row 2, column 1 block. Block is assigned upper object's label: " << getLabelUP << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowUp.y << " , " << windowUp.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUp, blockcolUp) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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
						cv::putText(Roi, str1_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						/*std::cout << "New object at pixels: " << windows.y << " , " << windows.x << endl;
						std::cout << "Object Number: " << objcounter << endl;
						std::cout << "Average distance of this object: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/

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

							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							//print new label of the object
							/*std::cout << "All three around similar. Label of any of them (left): " << getLabelLEFT << endl;
							std::cout << "Hence, block is assigned label: " << getLabelLEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/

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
							cv::putText(Roi, str_UP_UPLEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

							//print new label of the object
							/*std::cout << "All three around similar. Majority with label at UP/UPLEFT: " << getLabelUP_UPLEFT << endl;
							std::cout << "Hence, block is assigned label: " << getLabelUP_UPLEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowUpLeft.y << " , " << windowUpLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUpLeft, blockcolUpLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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

							cv::putText(Roi, str_UPLEFT_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


							//print new label of the object
							/*std::cout << "All three around similar. Majority with label at LEFT/UPLEFT: " << getLabelUPLEFT_LEFT << endl;
							std::cout << "Hence, block is assigned label: " << getLabelUPLEFT_LEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUpLeft, blockcolUpLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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
							cv::putText(Roi, str_UP_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//			cv::putText(Roi, str_UP_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);


							//print new label of the object
							/*std::cout << "All three around similar. Majority with label at LEFT/UP: " << getLabelUP_LEFT << endl;
							std::cout << "Hence, block is assigned label: " << getLabelUP_LEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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
							cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
							//			cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);


							//print new label of the object
							/*std::cout << "All three around similar. Lbls different. Assigned lbl of LEFT: " << getLabelLEFT << endl;
							std::cout << "Hence, block is assigned label: " << getLabelLEFT << endl;

							std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
							std::cout << endl;
							std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
							std::cout << endl;
							std::cout << endl;

							std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
							std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
							std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
							std::cout << endl;
							std::cout << endl;*/
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

						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						//print new label of the object
						/*std::cout << "All three around different. By default then assigned label of LEFT: " << getLabelLEFT << endl;
						std::cout << "Hence, block is assigned label: " << getLabelLEFT << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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
						cv::putText(Roi, str_Up_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
						//				cv::putText(Roi, str_Up_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);


						//print new label of the object
						/*std::cout << "All three around different. By default then assigned label of LEFT: " << getLabelUp << endl;
						std::cout << "Hence, block is assigned label: " << getLabelUp << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowUp.y << " , " << windowUp.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUp, blockcolUp) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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
						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);
						//					cv::putText(Roi, str_LEFT_detect.str(), cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cv::Scalar(255, 255, 255), 1, 8, false);


						//print new label of the object
						/*std::cout << "All three around different. By default then assigned label of LEFT: " << getLabelLEFT << endl;
						std::cout << "Hence, block is assigned label: " << getLabelLEFT << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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

						cv::putText(Roi, str_LEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


						//print new label of the object
						/*std::cout << "All three around different. By default then assigned label of LEFT: " << getLabelLEFT << endl;
						std::cout << "Hence, block is assigned label: " << getLabelLEFT << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowLeft.y << " , " << windowLeft.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowLeft, blockcolLeft) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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

						cv::putText(Roi, str_UPLEFT_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);

						//print new label of the object
						/*std::cout << "Similar to the UPLEFT block: " << getLabelUPLEFT << endl;
						std::cout << "Hence, block is assigned label: " << getLabelUPLEFT << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowUpLeft.y << " , " << windowUpLeft.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUpLeft, blockcolUpLeft) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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

						cv::putText(Roi, str_UP_label.str(), cv::Point(7, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1, 8, false);


						//print new label of the object
						/*std::cout << "Similar to the UPLEFT block: " << getLabelUP << endl;
						std::cout << "Hence, block is assigned label: " << getLabelUP << endl;

						std::cout << "Object at pixels: " << windows.y << " , " << windows.x;
						std::cout << endl;
						std::cout << "Similar to the one at pixels: " << windowUp.y << " , " << windowUp.x;
						std::cout << endl;
						std::cout << endl;

						std::cout << "Average distance of this BLOCK: " << block_matrix.get_average_distance_block_row_cols(blockrow, blockcol) << endl;
						std::cout << "Average distance of this OBJECT: " << block_matrix.get_average_distance_block_row_cols(blockrowUp, blockcolUp) << endl;
						std::cout << "Assigned label: " << block_matrix.get_label_block_row_cols(blockrow, blockcol) << endl;
						std::cout << endl;
						std::cout << endl;*/
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
	}//end segmenting()

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

#endif
	};