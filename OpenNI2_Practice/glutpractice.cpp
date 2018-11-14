
//General Headers
#include <stdio.h>
#include "pch.h"
#include <iostream>

//OpenNi2 Headers
#include <OpenNI.h>
using namespace openni;

//NiTE2 Headers
#include <NiTE.h>

//GLUT headers
#include<GL/glut.h>

/*
void gl_KeyboardCallBack(unsigned char key, int x, int y)
{
	if (key == 27)//esc key
	{
		depthSensor.destroy();
		OpenNI::shutdown();
		exit(0);
	}
}

//function to get continious data stream
void gl_IdleCallBack()
{
	glutPostRedisplay(); //this function gets the new frame and is called everytime the system idles
}


//function that will be called each time a opengl needs a new frame
void gl_DisplayCallback()
{
	if (depthSensor.isValid())
	{
		Status status = STATUS_OK;
		VideoStream* streamPointer = &depthSensor;
		int streamReadyIndex;
		status = OpenNI::waitForAnyStream(&streamPointer, 1, &streamReadyIndex, 500); //the 500 is a 500 ms timeout
		if (status == STATUS_OK && streamReadyIndex == 0)
		{
			VideoFrameRef newFrame;
			status = depthSensor.readFrame(&newFrame);
			if (status == STATUS_OK && newFrame.isValid())
			{
				//clear the opengl buffers
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				//setup the opengl viewpoint
				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				glOrtho(0, window_w, window_h, 0, -1.0, 1.0);

				//updating the texture (depth 1mm to rgb888)
				unsigned short maxDepth = 0;
				for (int y = 0; y < newFrame.getHeight(); ++y)
				{
					DepthPixel* depthCell = (DepthPixel*)((char*)newFrame.getData() + (y * newFrame.getStrideInBytes()));
					for (int x = 0; x < newFrame.getWidth(); ++x, ++depthCell)
					{
						if (maxDepth < *depthCell)
						{
							maxDepth = *depthCell;
						}
					}
				}

				//code to resize the incoming frame data to fit our declared window size. since window and texture buffer are the same size, the ratio will just be 1
				double resizeFactor = min((window_w / (double)newFrame.getWidth()), (window_h / (double)newFrame.getHeight()));
				unsigned int texture_x = (unsigned int)(window_w - (resizeFactor * newFrame.getWidth())) / 2;
				unsigned int texture_y = (unsigned int)(window_h - (resizeFactor * newFrame.getHeight())) / 2;

				//code to fill each pixel of our texture buffer
				for (unsigned int y = 0; y<(window_h - 2 *texture_y); ++y)
				{
					OniRGB888Pixel* texturePixel = gl_texture + ((y + texture_y)*window_w) + texture_x;
					for (unsigned int x = 0; x < (window_w - 2 * texture_x); ++x)
					{
						DepthPixel* streamPixel = (DepthPixel*)((char*)newFrame.getData() + ((int)(y / resizeFactor)*newFrame.getStrideInBytes())) + (int)(x / resizeFactor);//use char* to convert into a byte so the first addition is in bytes, and second is in pixels
						if (*streamPixel != 0)
						{
							//scales the value into a byte value i.e 0->255 so that further away pixels are darker versions of grey
							char depthValue = ((float)*streamPixel / maxDepth) * 255;
							texturePixel->b = 255 - depthValue;
							texturePixel->g = 255 - depthValue;
							texturePixel->r = 255 - depthValue;
						}
						else //objects that are a distance <0.5m or >5m shown as black
						{
							texturePixel->b = 0;
							texturePixel->g = 0;
							texturePixel->r = 0;
						}
						texturePixel += 1; //moves variable by 3 bytes
					}
				}
				//create the opengl texture map and positioning our texture
				glTexParameteri(GL_TEXTURE_2D, 0x8191, GL_TRUE); //0x8191 = GL_GENERATE_MIPMAP but glut did not define it
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_w, window_h, 0, GL_RGB, GL_UNSIGNED_BYTE, gl_texture);
				glBegin(GL_QUADS);
				glTexCoord2f(0.0f, 0.0f);
				glVertex3f(0.0f, 0.0f, 0.0f);
				glTexCoord2f(0.0f, 1.0f);
				glVertex3f(0.0f, (float)window_h, 0.0f);
				glTexCoord2f(1.0f, 1.0f);
				glVertex3f((float)window_w, (float)window_h, 0.0f);
				glTexCoord2f(1.0f, 0.0f);
				glVertex3f((float)window_w, 0.0f, 0.0f);
				glEnd();
				glutSwapBuffers();//moves current buffer to the front
			}
		}
	}
}
*/

int main2(int argc, char** argv[])
{
	printf("Hello World!\r\n");

	/*
	printf("Initializing OpenGL ...\r\n");
	gl_texture = (OniRGB888Pixel*)malloc(window_w*window_h * sizeof(OniRGB888Pixel));//requests required memory from RAM (640*480=307,200 pixels. each pixel is 3 bytes. so 1612800 bytes)
	//intitialise opengl
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);//forces opengl to use both the colour and depth buffer as well as double buffering
	//initialise opengl window
	glutInitWindowSize(window_w, window_h); // set size of window
	glutCreateWindow("Data Stream From Kinect"); //label the window
	//tells open gl which functions to use
	glutKeyboardFunc(gl_KeyboardCallBack);//declared earlier
	glutDisplayFunc(gl_DisplayCallback);//declared earlier
	glutIdleFunc(gl_IdleCallBack);//decared earlier
	//glutTimerFunc(1000, gl_IdleCallBack, NULL); //function to limit fps of program to 5
	//enable 2d display of texture, disable depth buffer updating, and start redering process
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	printf("Starting OpenGl rendering process ...\r\n");
	glutMainLoop();//locks our program in an infinite loop of opengl, any code after this will not be executed
	*/

	return 0;
}