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
#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H
using namespace std;

#include <opencv2/core.hpp>

//在命名空间中定义结构体
namespace DetectText {

	//某点的笔划宽度信息，坐标和宽度值
	struct SWTPoint2d {
		int x;
		int y;
		float SWT;
	};

	//命名别名，一对SWT点，一对点
	typedef std::pair<SWTPoint2d, SWTPoint2d> SWTPointPair2d;
	typedef std::pair<cv::Point, cv::Point>   SWTPointPair2i;

	//浮点数的点
	struct Point2dFloat{
		float x;
	    float y;
};

//梯度寻找路径的线段，起点，终点，中间经过的点
struct Ray {
	SWTPoint2d p;
	SWTPoint2d q;
	std::vector<SWTPoint2d> points;
};

//三维浮点数的点
struct Point3dFloat {
	float x;
	float y;
	float z;
};

//文本行
struct Chain {
	int p;
	int q;
	float dist;
	bool merged;
	Point2dFloat direction;
	std::vector<int> components;
};


//点排序
bool Point2dSort(SWTPoint2d const & lhs,
	SWTPoint2d const & rhs);

//文本检测
cv::Mat textDetection(const cv::Mat& input, bool dark_on_light);

//笔划宽度转换，从梯度图像得到笔划宽度图像和梯度寻找线段
void strokeWidthTransform(const cv::Mat& edgeImage,
	cv::Mat& gradientX,
	cv::Mat& gradientY,
	bool dark_on_light,
	cv::Mat& SWTImage,
	std::vector<Ray> & rays);

//笔划宽度过滤
void SWTMedianFilter(cv::Mat& SWTImage, std::vector<Ray> & rays);

//寻找有效连通域
std::vector< std::vector<SWTPoint2d> > findLegallyConnectedComponents(cv::Mat& SWTImage, std::vector<Ray> & rays);

//寻找有效连通域的梯度线段
std::vector< std::vector<SWTPoint2d> >
findLegallyConnectedComponentsRAY(cv::Mat& SWTImage, std::vector<Ray> & rays);

//连通域的状态
void componentStats(cv::Mat& SWTImage, const std::vector<SWTPoint2d> & component,
	float & mean, float & variance, float & median,
	int & minx, int & miny, int & maxx, int & maxy);

//连通域过滤
void filterComponents(cv::Mat& SWTImage,
	std::vector<std::vector<SWTPoint2d> > & components,
	std::vector<std::vector<SWTPoint2d> > & validComponents,
	std::vector<Point2dFloat> & compCenters,
	std::vector<float> & compMedians,
	std::vector<SWTPoint2d> & compDimensions,
	std::vector<SWTPointPair2d > & compBB);

//连成文本行
std::vector<Chain> makeChains(const cv::Mat& colorImage,
	std::vector<std::vector<SWTPoint2d> > & components,
	std::vector<Point2dFloat> & compCenters,
	std::vector<float> & compMedians,
	std::vector<SWTPoint2d> & compDimensions,
	std::vector<SWTPointPair2d > & compBB);

}

#endif 

