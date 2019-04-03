
#ifndef HeaderOlya_h
#define HeaderOlya_h

//Similar to DistanceBlockMatrix but with average distance being Scalar rather than short...
#include <Windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

////////////////////   ======================================================

// at the end thre is a test main to check it all works


// class used to hold information about a block
class distance_block {
protected:
	// block coordintates in the original image
	int x0, y0;

	// block dimensions in the original image
	int width, height;

	Scalar average_distance;

	short label;

	//flag to say block wasnt homogenous 
	bool homflag;

public:

	// constructor. if nothing assigned- these values will be  -o
	distance_block()
	{
		label = 99; average_distance = -1;
		x0 = -1; y0 = -1;
		width = 0; height = 0;
		homflag = true;
	}
	Scalar get_average_distance() { return average_distance; }

	void set_average_distance(Scalar inp_average_distance)
	{
		average_distance = inp_average_distance;
	}

	////////////////////OLYA////////////////////

	void set_label(short inp_label) {
		label = inp_label;
	}
	////////////////////////////////////////////

	void set_coordinates(int inp_x0, int inp_y0) { x0 = inp_x0; y0 = inp_y0; }
	int get_coordinate_x0() { return x0; }
	int get_coordinate_y0() { return y0; }
	void set_dimensions(int inp_width, int inp_height) { width = inp_width; height = inp_height; }
	int get_width() { return width; }
	int get_height() { return height; }
	short get_label() { return label; }

	void set_homflag(bool flag) { homflag = flag; }

	bool get_homflag() { return homflag; }
};


// class used to hold informtion about all blocks in an image
class distance_matrix {
protected:

	// image size
	int image_width, image_height;

	short result1, result2;
	Scalar output;

	static const int default_image_width = 640;
	static const int default_image_height = 480;

	// blocks in the image
	int block_rows, block_cols;

	//if not integer number of row/cols (for example, 10.7 rows) -o
	int last_row_height, last_col_width;

	// block size
	int block_width, block_height;
	//default those that I originally created 
	static const int default_block_width = 10;
	static const int default_block_height = 10;

	//matrix - pointer of pointers (big box with another box inside with things in it) -o
	distance_block** matrix;

	//need to allocate in memory for its proper use -o
	bool matrixm_allocated;

	//array of distance blocks (with their dimensions and labels ...) -o
	distance_block* linear_array;
	bool linear_array_allocated;

	//allocate matrix in memory (matrix that contains pointers to blocks (with labels, rows, cols etc))
	bool matrix_allocate()
	{
		//check why use 'new' etc???????-o
		linear_array = new distance_block[block_rows*block_cols];
		if (linear_array == NULL)
			return false;

		linear_array_allocated = true;


		/*
		//check why use 'calloc' etc?????????-o
		matrix = (distance_block**)calloc(block_rows, sizeof(distance_block*));
		if (linear_array == NULL)
		return false;
		*/
		//check why use 'calloc' etc?????????-o
		matrix = (distance_block**)calloc(block_rows, sizeof(distance_block*));
		if (matrix == NULL)
			return false;

		matrixm_allocated = true;

		// fill the matrix with the correct pointers (correct addresses ???//) -o
		distance_block* startaddress = linear_array;

		for (int i = 0; i<block_rows; i++)
		{
			matrix[i] = startaddress + (i*block_cols);
		}


		return (matrixm_allocated && linear_array_allocated);

	}

	void matrix_setup()
	{
		if (matrixm_allocated && linear_array_allocated)
		{
			// setup each block in the matrix
			for (int i = 0; i < block_rows; i++) {
				for (int j = 0; j < block_cols; j++)
				{
					// set block coordinates
					//I assume, x0 y0 are bottom-right coordinates as you multiply i,j by height and width  -o
					int x0 = i*block_height;
					int y0 = j*block_width;
					//set coorditanes of all blocks !!!
					matrix[i][j].set_coordinates(x0, y0);

					// set block dimensions
					int inp_width, inp_height;
					if (i == block_rows - 1 && last_row_height > 0)
						inp_height = last_row_height;
					else
						inp_height = block_height;
					if (j == block_cols - 1 && last_col_width > 0)
						inp_width = last_col_width;
					else
						inp_width = block_width;
					//if last block column/row not 60 & 60, set its dimensions in matrix at correspondent i,j! -o
					matrix[i][j].set_dimensions(inp_width, inp_height);
				}
			}
		}

		//Olya:
		else {
			cout << "Error!!! Either array or matrix not allocated. Noe entering the loop!";
			cout << endl;
		}

	}

public:
	distance_matrix()
	{
		matrix = NULL;
		linear_array = NULL;
		matrixm_allocated = false;
		linear_array_allocated = false;
		block_rows = 0;
		block_cols = 0;
	}
	~distance_matrix()
	{
		// Distructor: deallocate the memory if it was allocated;
		cout << endl << "Matrix will be deleted" << endl;

		if (linear_array_allocated)
			delete linear_array; // allocated with "new": deallocate with "delete"

		if (matrixm_allocated)
			free(matrix); // allocated with "calloc": deallocate with "free()"

		cout << endl << "Matrix deleted" << endl;

	}

	int get_block_rows() { return block_rows; }
	int get_block_cols() { return block_cols; }

	//function to set whether a block is homogenous or not
	void set_block_homflag_row_cols(int inp_block_row, int inp_block_col, bool flag)
	{
		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row < block_rows)
				if (inp_block_col >= 0 && inp_block_col < block_cols)
					// for block with these row-column set the average distance

					matrix[inp_block_row][inp_block_col].set_homflag(flag);
				else
					cout << endl << "inp_block_col=" << inp_block_col << " exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << " exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;
	}

	//returns the value to say whether a bloc is homogenous
	bool get_block_homflag_row_cols(int inp_block_row, int inp_block_col)
	{
		bool flag;
		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row < block_rows)
				if (inp_block_col >= 0 && inp_block_col < block_cols)
					// for block with these row-column set the average distance
					//SET THE LABEL FOR THE SINGLE BLOCK. SET_LABEL FUNCTION IS PRESENT IN BLOCK CLASS!!!
					//here is the function to do it for all the blocks depending on the row/column
					//short result added *******
					flag = matrix[inp_block_row][inp_block_col].get_homflag();

				else
					cout << endl << "inp_block_col=" << inp_block_col << "exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << "exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

		return flag;
	}

	bool setup_matrix(int inp_image_width, int inp_image_height, int inp_block_width, int inp_block_height)
	{
		bool memoryresult = false;


		// do it only once, when the memory has not been allocated yet
		// if this function is called a second time (on a matrix that has alredy been allocated) it does nothing
		if ((!matrixm_allocated) && (!linear_array_allocated))
		{


			// check input values and use default values if any parameter is non-positive
			if (inp_image_width <= 0)
				inp_image_width = default_image_width;
			if (inp_image_height <= 0)
				inp_image_height = default_image_height;
			if (inp_block_width <= 0)
				inp_block_width = default_block_width;
			if (inp_block_height <= 0)
				inp_block_height = default_block_height;

			// typical block size
			block_width = inp_block_width;
			block_height = inp_block_height;


			// rows; since rows are int, the result will be forced to the integer value (if 10.7 rows, for example)-o
			block_rows = inp_image_height / inp_block_height;
			//if 10.7 rows, last row height will be calculated and stored, and the number of blocks will be 11 
			//(with the last being small) -o
			last_row_height = inp_image_height - (block_rows*inp_block_height);
			if (last_row_height>0)
				block_rows++;
			//

			// cols
			block_cols = inp_image_width / inp_block_width;
			last_col_width = inp_image_width - (block_cols*inp_block_width);
			if (last_col_width>0)
				block_cols++;
			//Olya--------------------------------------------------------------------
			//cout << "Block Rows here: " << block_rows;
			//cout << "Block columns here: " << block_cols;
			//cout << "Dont forget to automize it ......" << endl;
			//cout << endl;
			//------------------------------------------------------------------------


			// allocate memory (call function from protected)

			memoryresult = matrix_allocate();
			//memoryresult = true;


			if (memoryresult)
			{
				// populate the matrix ; set dimensions, coordinates and height/width of last blocks if they are of an odd size -o
				matrix_setup();
			}
			else
			{
				// print error message: out of memory with input values....;
				cout << endl << "Out of MEM" << endl;
			}
		}

		return memoryresult;
	}
	//since when you set 0's it forces values to be default (width 60 for example)
	bool setup_matrix_defaultvals() { return setup_matrix(0, 0, 0, 0); }

	// this function sets the average distrance to a block with given row and cols index
	void set_average_distance_block_row_cols(int inp_block_row, int inp_block_col, Scalar inp_average_distance)
	{

		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row<block_rows)
				if (inp_block_col >= 0 && inp_block_col<block_cols)
					// for block with these row-column set the average distance

					matrix[inp_block_row][inp_block_col].set_average_distance(inp_average_distance);
				else
					cout << endl << "inp_block_col=" << inp_block_col << " exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << " exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

	}

	// same as above, starting with the coordinate of a point inside the block
	void set_average_distance_block_coordinates(int inp_x, int inp_y, Scalar inp_average_distance)
	{
		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_x >= 0 && inp_x<image_height)
				if (inp_y >= 0 && inp_x<image_width)
				{
					int inp_block_row = image_height / block_height;
					int inp_block_col = image_width / block_width;
					set_average_distance_block_row_cols(inp_block_row, inp_block_col, inp_average_distance);
				}
				else
					cout << endl << "inp_x=" << inp_x << " exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_y=" << inp_y << " exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;
	}

	// ---------------------------------------------------------------------------------------------------------------------
	/////////////OLYA///////////////////////
	Scalar get_average_distance_block_row_cols(int inp_block_row, int inp_block_col)
	{

		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row<block_rows)
				if (inp_block_col >= 0 && inp_block_col<block_cols)
					// for block with these row-column set the average distance

					output = matrix[inp_block_row][inp_block_col].get_average_distance();
				else
					cout << endl << "inp_block_col=" << inp_block_col << "exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << "exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

		return output;

	}

	// this function sets the label to a block with given row and cols index
	void set_label_block_row_cols(int inp_block_row, int inp_block_col, short inp_label)
	{

		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row<block_rows)
				if (inp_block_col >= 0 && inp_block_col<block_cols)
					// for block with these row-column set the average distance
					//SET THE LABEL FOR THE SINGLE BLOCK. SET_LABEL FUNCTION IS PRESENT IN BLOCK CLASS!!!
					//here is the function to do it for all the blocks depending on the row/column
					matrix[inp_block_row][inp_block_col].set_label(inp_label);
				else
					cout << endl << "inp_block_col=" << inp_block_col << " exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << " exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

	}

	void reset_blocks()
	{
		for (int row = 0; row < block_rows; row++)
		{
			for (int col = 0; col < block_cols; col++)
			{
				matrix[row][col].set_average_distance(0);
				matrix[row][col].set_label(0);
			}
		}
	}

	// same as above, starting with the coordinate of a point inside the block
	void set_label_block_coordinates(int inp_x, int inp_y, short inp_label)
	{
		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_x >= 0 && inp_x<image_height)
				if (inp_y >= 0 && inp_x<image_width)
				{
					int inp_block_row = image_height / block_height;
					int inp_block_col = image_width / block_width;
					set_label_block_row_cols(inp_block_row, inp_block_col, inp_label);
				}
				else
					cout << endl << "inp_x=" << inp_x << " exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_y=" << inp_y << " exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;
	}

	//uncomment
	short get_label_block_row_cols(int inp_block_row, int inp_block_col)
	{

		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_block_row >= 0 && inp_block_row<block_rows)
				if (inp_block_col >= 0 && inp_block_col<block_cols)
					// for block with these row-column set the average distance
					//SET THE LABEL FOR THE SINGLE BLOCK. SET_LABEL FUNCTION IS PRESENT IN BLOCK CLASS!!!
					//here is the function to do it for all the blocks depending on the row/column
					//short result added *******
					result1 = matrix[inp_block_row][inp_block_col].get_label();

				else
					cout << endl << "inp_block_col=" << inp_block_col << "exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_block_row=" << inp_block_row << "exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

		return result1;
	}

	//here uncomment
	short get_label_block_coordinates(int inp_x, int inp_y)
	{
		if ((matrixm_allocated) && (linear_array_allocated))
			if (inp_x >= 0 && inp_x < image_height)
				if (inp_y >= 0 && inp_x < image_width)
				{
					int inp_block_row = image_height / block_height;
					int inp_block_col = image_width / block_width;
					result2 = get_label_block_row_cols(inp_block_row, inp_block_col);
				}
				else
					cout << endl << "inp_x=" << inp_x << "exceeds matrix dimension" << endl;
			else
				cout << endl << "inp_y=" << inp_y << "exceeds matrix dimension" << endl;
		else
			cout << endl << "Matrix not allocated" << endl;

		return result2;
	}

	//--------------------------------------------------------------------------------------------------------------------------

	// use during debug to check the matrix has been filled correctly
	void printToScreen()
	{
		if (matrixm_allocated && linear_array_allocated)
		{
			cout << endl << "  --->  " << block_rows << " x " << block_cols << " blocks in the matrix. Printing blocks row-by-row:" << endl << endl;

			// Print to screen the content of each block in the matrix
			for (int i = 0; i<block_rows; i++)
			{
				for (int j = 0; j<block_cols; j++)
				{
					// get block coordinates
					/*
					int x0 = matrix[i][j].get_coordinate_x0();
					int y0 = matrix[i][j].get_coordinate_y0();
					int blkwidth = matrix[i][j].get_width();
					int blkheight = matrix[i][j].get_height();
					cout << "Block (i=" << i << " , j=" << j << ") coordinates are (x0=" << x0 << " , y0=" << y0 << ");";
					cout << " width=" << blkwidth << " ; height=" << blkheight << " .";
					cout << " Distance=" << matrix[i][j].get_average_distance() << " ; label=" << matrix[i][j].get_label() << " ;";
					cout << endl;
					*/
					//char c = matrix[i][j].get_label();
					cout << left;
					cout <<setw(2)<< matrix[i][j].get_label() << "";
					//cout << c;
				}
				cout << endl;
			}
		}
		else
		{
			// print error message: out of memory with input values....;
			cout << endl << "Empty marix: nothing to print" << endl << endl;
		}
	}

};


// this is a test main to check it all works
/*
void main()
{
{
distance_matrix testmatrix;
//testmatrix.printToScreen();


testmatrix.setup_matrix_defaultvals();
testmatrix.printToScreen();
}

cout << endl << "test done" << endl;
cout << endl << endl;
}
//*/
////////////////////   ======================================================


#endif