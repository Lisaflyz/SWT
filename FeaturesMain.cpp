/*
Copyright 2012 Andrew Perrault and Saurav Kumar.

This file is part of DetectText.

DetectText is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DetectText is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cassert>
#include <fstream>
#include <exception>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "TextDetection.h"

using namespace std;
using namespace cv;
using namespace DetectText;

//位图转换为浮点图的过程
void convertToFloatImage(Mat& byteImage, Mat& floatImage)
{
	byteImage.convertTo(floatImage, CV_32FC1, 1 / 255.);
}

//错误情况
class FeatureError : public std::exception {
	std::string message;
public:
	FeatureError(const std::string & msg, const std::string & file) {
		std::stringstream ss;

		ss << msg << " " << file;
		message = msg.c_str();
	}
	~FeatureError() throw () {
	}
};

//载入位图的过程
Mat loadByteImage(const char * name) {
	Mat image = imread(name);

	if (image.empty()) {
		return Mat();
	}
	cvtColor(image, image, COLOR_BGR2RGB);
	return image;
}

//载入浮点图
Mat loadFloatImage(const char * name) {
	Mat image = imread(name);

	if (image.empty()) {
		return Mat();
	}
	cvtColor(image, image, COLOR_BGR2RGB);
	Mat floatingImage(image.size(), CV_32FC3);
	image.convertTo(floatingImage, CV_32F, 1 / 255.);
	return floatingImage;
}

int mainTextDetection(int argc, char** argv) {
	//载入位图
	
	Mat byteQueryImage = imread("E:/project/data/test/img_6.jpg");
	if (byteQueryImage.empty()) {//载入的图片为空，则退出
		cerr << "couldn't load query image" << endl;
		return -1;
	}

	// 在图片中检测文本，参数3为将字符串转换成整型数， dark_on_light为正数

	double start = (double)getTickCount();
	Mat output = textDetection(byteQueryImage, atoi(argv[3]));
	double duration = ((double)getTickCount() - start) / getTickFrequency();
	cout << "------------------检测所花的时间为： " << duration << " 秒" << endl;
	//将检测到的图片写入到参数2所指的文件
	imwrite(argv[2], output);
	imshow("原图", byteQueryImage);
	imshow("结果图", output);
	
	waitKey(0);
	return 0;
}

int main(int argc, char** argv) {
	if ((argc != 4)) {
		cerr << "usage: " << argv[0] << " imagefile resultImage darkText" << endl;
		return -1;
	}
	//参数符合条件，调用方法
	return mainTextDetection(argc, argv);
}
