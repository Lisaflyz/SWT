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
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>

using namespace cv;

#include "TextDetection.h"

#define PI 3.14159265

namespace DetectText {

	const Scalar BLUE(255, 0, 0);
	const Scalar GREEN(0, 255, 0);
	const Scalar RED(0, 0, 255);

	std::vector<SWTPointPair2i > findBoundingBoxes(std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<Chain> & chains,
		std::vector<SWTPointPair2d > & compBB,
		Mat& output) {
		std::vector<SWTPointPair2i > bb;
		bb.reserve(chains.size());
		for (auto& chainit : chains) {
			int minx = output.cols;
			int miny = output.rows;
			int maxx = 0;
			int maxy = 0;
			for (std::vector<int>::const_iterator cit = chainit.components.begin(); cit != chainit.components.end(); cit++) {
				miny = std::min(miny, compBB[*cit].first.y);
				minx = std::min(minx, compBB[*cit].first.x);
				maxy = std::max(maxy, compBB[*cit].second.y);
				maxx = std::max(maxx, compBB[*cit].second.x);
			}
			Point2i p0(minx, miny);
			Point2i p1(maxx, maxy);
			SWTPointPair2i pair(p0, p1);
			bb.push_back(pair);
		}
		return bb;
	}

	std::vector<SWTPointPair2i > findBoundingBoxes(std::vector<std::vector<SWTPoint2d> > & components,
		Mat& output) {
		std::vector<SWTPointPair2i > bb;
		bb.reserve(components.size());
		for (auto& compit : components) {
			int minx = output.cols;
			int miny = output.rows;
			int maxx = 0;
			int maxy = 0;
			for (auto& it : compit) {
				miny = std::min(miny, it.y);
				minx = std::min(minx, it.x);
				maxy = std::max(maxy, it.y);
				maxx = std::max(maxx, it.x);
			}
			Point2i p0(minx, miny);
			Point2i p1(maxx, maxy);
			SWTPointPair2i pair(p0, p1);
			bb.push_back(pair);
		}
		return bb;
	}

	//图片归一化
	void normalizeImage(const Mat& input, Mat& output) {
		assert(input.depth() == CV_32F);
		assert(input.channels() == 1);
		assert(output.depth() == CV_32F);
		assert(output.channels() == 1);

		float maxVal = 0;
		float minVal = 1e100;
		//找到整张图片中的最大值和最小值
		for (int row = 0; row < input.rows; row++){
			const float* ptr = (const float*)input.ptr(row);
			for (int col = 0; col < input.cols; col++){
				if (*ptr < 0) {}
				else {
					maxVal = std::max(*ptr, maxVal);
					minVal = std::min(*ptr, minVal);
				}
				ptr++;
			}
		}

		//小于0的点（背景点赋为1），其余点求百分比
		float difference = maxVal - minVal;
		for (int row = 0; row < input.rows; row++) {
			const float* ptrin = (const float*)input.ptr(row);
			float* ptrout = (float*)output.ptr(row);
			for (int col = 0; col < input.cols; col++) {
				if (*ptrin < 0) {
					*ptrout = 1;
				}
				else {
					*ptrout = ((*ptrin) - minVal) / difference;
				}
				ptrout++;
				ptrin++;
			}
		}
	}

	//画出字符连通域
	void renderComponents(const Mat& SWTImage, std::vector<std::vector<SWTPoint2d> > & components, Mat& output) {
		output.setTo(0);

		//拷贝SWT图的有效值
		for (auto& component : components) {
			for (auto& pit : component) {
				output.at<float>(pit.y, pit.x) = SWTImage.at<float>(pit.y, pit.x);
			}
		}

		//无SWT有效值的区域置为-1
		for (int row = 0; row < output.rows; row++){
			float* ptr = (float*)output.ptr(row);
			for (int col = 0; col < output.cols; col++){
				if (*ptr == 0) {
					*ptr = -1;
				}
				ptr++;
			}
		}

		//取得SWT的最大值和最小值
		float maxVal = 0;
		float minVal = 1e100;
		for (int row = 0; row < output.rows; row++){
			const float* ptr = (const float*)output.ptr(row);
			for (int col = 0; col < output.cols; col++){
				if (*ptr == 0) {}
				else {
					maxVal = std::max(*ptr, maxVal);
					minVal = std::min(*ptr, minVal);
				}
				ptr++;
			}
		}

		//得到极差，归一化
		float difference = maxVal - minVal;
		for (int row = 0; row < output.rows; row++){
			float* ptr = (float*)output.ptr(row);
			for (int col = 0; col < output.cols; col++){
				if (*ptr < 1) {
					*ptr = 1;
				}
				else {
					*ptr = ((*ptr) - minVal) / difference;
				}
				ptr++;
			}
		}

	}

	//对过滤后的每个连通域标出矩形框
	void renderComponentsWithBoxes(Mat& SWTImage, std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<SWTPointPair2d > & compBB, Mat& output) {
		Mat outTemp(output.size(), CV_32FC1);//得到副本，用于改变和重绘

		//对字符连通域归一化表示
		renderComponents(SWTImage, components, outTemp);

		//复制每个连通域的最小值和最大值
		std::vector<SWTPointPair2i> bb;
		bb.reserve(compBB.size());
		for (auto& it : compBB) {
			Point2i p0 = Point(it.first.x, it.first.y);
			Point2i p1 = Point(it.second.x, it.second.y);
			SWTPointPair2i pair(p0, p1);
			bb.push_back(pair);
		}

		Mat out(output.size(), CV_8UC1);
		//255为scale factor
		outTemp.convertTo(out, CV_8UC1, 255.);
		cvtColor(out, output, COLOR_GRAY2RGB);

		//以红蓝绿交替画出矩形框，包围字符
		int count = 0;
		for (auto it : bb) {
			Scalar c;
			if (count % 3 == 0) {
				c = BLUE;
			}
			else if (count % 3 == 1) {
				c = GREEN;
			}
			else {
				c = RED;
			}
			count++;
			//线宽为两个像素
			rectangle(output, it.first, it.second, c, 2);
		}
	}

	void renderChainsWithBoxes(Mat& SWTImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<Chain> & chains,
		std::vector<SWTPointPair2d > & compBB,
		Mat& output) {
		// keep track of included components
		std::vector<bool> included;
		included.reserve(components.size());
		for (unsigned int i = 0; i != components.size(); i++) {
			included.push_back(false);
		}
		for (Chain& it : chains) {
			for (std::vector<int>::iterator cit = it.components.begin(); cit != it.components.end(); cit++) {
				included[*cit] = true;
			}
		}
		std::vector<std::vector<SWTPoint2d> > componentsRed;
		for (unsigned int i = 0; i != components.size(); i++) {
			if (included[i]) {
				componentsRed.push_back(components[i]);
			}
		}
		Mat outTemp(output.size(), CV_32FC1);

		std::cout << componentsRed.size() << " components after chaining" << std::endl;
		renderComponents(SWTImage, componentsRed, outTemp);
		std::vector<SWTPointPair2i > bb;
		bb = findBoundingBoxes(components, chains, compBB, outTemp);

		Mat out(output.size(), CV_8UC1);
		outTemp.convertTo(out, CV_8UC1, 255);
		cvtColor(out, output, COLOR_GRAY2RGB);

		int count = 0;
		for (auto& it : bb) {
			Scalar c;
			if (count % 3 == 0) {
				c = BLUE;
			}
			else if (count % 3 == 1) {
				c = GREEN;
			}
			else {
				c = RED;
			}
			count++;
			rectangle(output, it.first, it.second, c, 2);
		}
	}

	//画出文本行
	void renderChains(Mat& SWTImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<Chain> & chains,
		Mat& output) {
		// keep track of included components
		std::vector<bool> included;
		included.reserve(components.size());
		for (unsigned int i = 0; i != components.size(); i++) {
			included.push_back(false);
		}
		for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end(); it++) {
			for (std::vector<int>::iterator cit = it->components.begin(); cit != it->components.end(); cit++) {
				included[*cit] = true;
			}
		}
		std::vector<std::vector<SWTPoint2d> > componentsRed;
		for (unsigned int i = 0; i != components.size(); i++) {
			if (included[i]) {
				componentsRed.push_back(components[i]);
			}
		}
		std::cout << componentsRed.size() << " components after chaining" << std::endl;
		Mat outTemp(output.size(), CV_32FC1);
		renderComponents(SWTImage, componentsRed, outTemp);
		outTemp.convertTo(output, CV_8UC1, 255);

	}



	//对三通道RGB图像进行文字检测
	Mat textDetection(const Mat& input, bool dark_on_light) {
		assert(input.depth() == CV_8U);
		assert(input.channels() == 3);

		std::cout << "Running textDetection with dark_on_light " << dark_on_light << std::endl;

		// 得到原图的灰度图
		Mat grayImage(input.size(), CV_8UC1);
		cvtColor(input, grayImage, COLOR_BGR2GRAY);
		// Canny边缘检测得到边缘图像,边缘点的值为255，其余值为0
		double threshold_low = 175;
		double threshold_high = 320;
		Mat edgeImage(input.size(), CV_8UC1);
		Canny(grayImage, edgeImage, threshold_low, threshold_high, 3);
		imwrite("canny.png", edgeImage);

		// 计算x和y方向上的梯度
		Mat gaussianImage(input.size(), CV_32FC1);
		grayImage.convertTo(gaussianImage, CV_32FC1, 1. / 255.);
		GaussianBlur(gaussianImage, gaussianImage, Size(5, 5), 0);
		Mat gradientX(input.size(), CV_32FC1);
		Mat gradientY(input.size(), CV_32FC1);
		// void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
	//与原图像深度一致，x方向上梯度
		Scharr(gaussianImage, gradientX, -1, 1, 0);
		Scharr(gaussianImage, gradientY, -1, 0, 1);
		GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
		GaussianBlur(gradientY, gradientY, Size(3, 3), 0);

		//计算SWT，返回梯度线段矢量
		std::vector<Ray> rays;
		Mat SWTImage(input.size(), CV_32FC1);
		//SWT图像中每个点赋初值为-1
		for (int row = 0; row < input.rows; row++){
			float* ptr = (float*)SWTImage.ptr(row);
			for (int col = 0; col < input.cols; col++){
				*ptr++ = -1;
			}
		}
		//得到笔划宽度变化后的图像SWTImage
		strokeWidthTransform(edgeImage, gradientX, gradientY, dark_on_light, SWTImage, rays);
		//对所有未舍弃的线段，计算其笔划中值，赋给大于中值的像素点，校正拐角处像素点的笔划宽度
		SWTMedianFilter(SWTImage, rays);

		//笔划宽度值转换为灰度图输出显示
		Mat output2(input.size(), CV_32FC1);
		normalizeImage(SWTImage, output2);
		Mat saveSWT(input.size(), CV_8UC1);
		output2.convertTo(saveSWT, CV_8UC1, 255);
		imwrite("SWT.png", saveSWT);



		//由SWT和梯度图像计算有效连通域，即字符区域，返回值为vector<vector>，
		//外部的vector代表连通域，内部的vector连通域中的（y,x）
		//用SWTImage和rays寻找有效连通域
		std::vector<std::vector<SWTPoint2d> > components = findLegallyConnectedComponents(SWTImage, rays);
	/*	Mat comMat(input.size(), CV_8UC1);
		for (int row = 0; row < input.rows; row++){
			int *ptr = (int*)comMat.ptr(row);
			for (int col = 0; col < comMat.cols; col++){
				*ptr++ = 200;
			}
		}

		
		for (int i = 0; i < components.size(); i++){
			for (int j = 0; j < components[i].size(); j++){
				SWTPoint2d temp = components[i][j];
				comMat.at<uchar>(i,j) = ++i;
			}
		}
*/
		// 过滤连通域
		std::vector<std::vector<SWTPoint2d> > validComponents;//连通域的集合
		std::vector<SWTPointPair2d > compBB;//某个连通域
		std::vector<Point2dFloat> compCenters;//连通域的中点
		std::vector<float> compMedians;
		std::vector<SWTPoint2d> compDimensions;
		//过滤连通域，根据自定义的规则滤除非文字区域
		filterComponents(SWTImage, components, validComponents, compCenters, compMedians, compDimensions, compBB);

		//用Bounging box框出归一化后的字符连通域，并画出
		Mat output3(input.size(), CV_8UC3);
		renderComponentsWithBoxes(SWTImage, validComponents, compBB, output3);
		imwrite("components.png", output3);
		

		// Make chains of components
		//将连通域组成文本行
		std::vector<Chain> chains;
		chains = makeChains(input, validComponents, compCenters, compMedians, compDimensions, compBB);

		Mat output4(input.size(), CV_8UC1);
		renderChains(SWTImage, validComponents, chains, output4);
		imwrite ( "text.png", output4);

		Mat output5(input.size(), CV_8UC3);
		cvtColor(output4, output5, COLOR_GRAY2RGB);
		return output5;
	}

	//笔划宽度转换，笔划拐点未处理
	void strokeWidthTransform(const Mat& edgeImage,
		Mat& gradientX,
		Mat& gradientY,
		bool dark_on_light,
		Mat& SWTImage,
		std::vector<Ray> & rays) {
		// First pass
		float prec = .05;//前进步长为0.05
		for (int row = 0; row < edgeImage.rows; row++){
			const uchar* ptr = (const uchar*)edgeImage.ptr(row);
			for (int col = 0; col < edgeImage.cols; col++){
				if (*ptr > 0) {//该点为白色边缘
					Ray r;

					SWTPoint2d p;
					p.x = col;
					p.y = row;
					r.p = p;
					std::vector<SWTPoint2d> points;
					points.push_back(p);

					//为什么要加0.5呢？
					float curX = (float)col + 0.5;
					float curY = (float)row + 0.5;
					int curPixX = col;
					int curPixY = row;
					float G_x = gradientX.at<float>(row, col);
					float G_y = gradientY.at<float>(row, col);
					// normalize gradient
					// 得到起始点的梯度
					float mag = sqrt((G_x * G_x) + (G_y * G_y));
					//朝着字的方向
					if (dark_on_light){
						G_x = -G_x / mag;
						G_y = -G_y / mag;
					}
					else {
						G_x = G_x / mag;
						G_y = G_y / mag;

					}
					while (true) {
						//以梯度强度的0.05倍往前
						curX += G_x*prec;
						curY += G_y*prec;
						//不为当前像素点，说明已前进到其它像素
						if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) {
							curPixX = (int)(floor(curX));
							curPixY = (int)(floor(curY));
							// 当前像素超出图片范围则停止
							if (curPixX < 0 || (curPixX >= SWTImage.cols) || curPixY < 0 || (curPixY >= SWTImage.rows)) {
								break;
							}
							SWTPoint2d pnew;
							pnew.x = curPixX;
							pnew.y = curPixY;
							points.push_back(pnew);

							//到达另一边缘，计算梯度
							if (edgeImage.at<uchar>(curPixY, curPixX) > 0) {
								r.q = pnew;
								// dot product
								float G_xt = gradientX.at<float>(curPixY, curPixX);
								float G_yt = gradientY.at<float>(curPixY, curPixX);
								mag = sqrt((G_xt * G_xt) + (G_yt * G_yt));
								if (dark_on_light) {
									G_xt = -G_xt / mag;
									G_yt = -G_yt / mag;
								}
								else {
									G_xt = G_xt / mag;
									G_yt = G_yt / mag;

								}

								//起点和终点的梯度范围差在一定范围内则有效，否则该路径无效
								if (acos(G_x * -G_xt + G_y * -G_yt) < PI / 2.0) {
									//计算梯度线段长度，即笔划宽度，更新该线段路径上的点的笔划宽度值
									float length = sqrt(((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + 
										((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
									for (std::vector<SWTPoint2d>::iterator pit = points.begin(); pit != points.end(); pit++) {
										if (SWTImage.at<float>(pit->y, pit->x) < 0) {
											SWTImage.at<float>(pit->y, pit->x) = length;
										}
										else {
											SWTImage.at<float>(pit->y, pit->x) = std::min(length, SWTImage.at<float>(pit->y, pit->x));
										}
									}
									r.points = points;
									rays.push_back(r);
								}
								break;
							}
						}
					}
				}
				ptr++;//若不是边缘点则往后寻找，或某一边缘点已处理完成
			}
		}

	}

	//对所有未舍弃的线段，计算其笔划中值，赋给大于中值的像素点，校正拐角处像素点的笔划宽度
	void SWTMedianFilter(Mat& SWTImage, std::vector<Ray> & rays) {
		for (auto& rit : rays) {
			for (auto& pit : rit.points) {//给SWTImage赋值
				pit.SWT = SWTImage.at<float>(pit.y, pit.x);
			}
			//线段路径上的点按笔划宽度从小到大排序，取中间点的笔划宽度，更新大于中间点的宽度
			//函数的引用
			std::sort(rit.points.begin(), rit.points.end(), &Point2dSort);
			float median = (rit.points[rit.points.size() / 2]).SWT;
			//对路径上的所有点进行比较更新
			for (auto& pit : rit.points) {
				SWTImage.at<float>(pit.y, pit.x) = std::min(pit.SWT, median);
			}
		}
	}

	bool Point2dSort(const SWTPoint2d &lhs, const SWTPoint2d &rhs) {
		return lhs.SWT < rhs.SWT;
	}

	/*——————————————————————————————寻找有效连通域即字符区域——————————————————————————————*/
	std::vector< std::vector<SWTPoint2d> > findLegallyConnectedComponents(Mat& SWTImage, std::vector<Ray> & rays) {
		boost::unordered_map<int, int> map;//key为笔划点像素位置，value为像素点标号
		boost::unordered_map<int, SWTPoint2d> revmap;//key为像素点标号，value为记录SWT点

		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
		int num_vertices = 0;
		// Number vertices for graph.  Associate each point with number
		// 遍历SWT图，将每个有效点标号
		for (int row = 0; row < SWTImage.rows; row++){
			float * ptr = (float*)SWTImage.ptr(row);
			for (int col = 0; col < SWTImage.cols; col++){
				if (*ptr > 0) {//非负的点，即有笔划宽度的点才标号
					map[row * SWTImage.cols + col] = num_vertices;
					SWTPoint2d p;
					p.x = col;
					p.y = row;
					revmap[num_vertices] = p;
					num_vertices++;
				}
				ptr++;
			}
		}

		Graph g(num_vertices); //整张图有多少有效的SWT点

		for (int row = 0; row < SWTImage.rows; row++){
			float * ptr = (float*)SWTImage.ptr(row);
			for (int col = 0; col < SWTImage.cols; col++){
				if (*ptr > 0) {
					//检查右边点，右下点，下边点，左下点与当前点的比值是否在一定范围内，
					//是则归于一组，即一个连通域
					int this_pixel = map[row * SWTImage.cols + col];
					if (col + 1 < SWTImage.cols) {//右边点
						float right = SWTImage.at<float>(row, col + 1);
						if (right > 0 && ((*ptr) / right <= 5.0 || right / (*ptr) <= 5.0))
							boost::add_edge(this_pixel, map.at(row * SWTImage.cols + col + 1), g);
					}
					if (row + 1 < SWTImage.rows) {
						if (col + 1 < SWTImage.cols) {
							float right_down = SWTImage.at<float>(row + 1, col + 1);
							if (right_down > 0 && ((*ptr) / right_down <= 5.0 || right_down / (*ptr) <= 5.0))
								boost::add_edge(this_pixel, map.at((row + 1) * SWTImage.cols + col + 1), g);
						}
						float down = SWTImage.at<float>(row + 1, col);
						if (down > 0 && ((*ptr) / down <= 5.0 || down / (*ptr) <= 5.0))
							boost::add_edge(this_pixel, map.at((row + 1) * SWTImage.cols + col), g);
						if (col - 1 >= 0) {
							float left_down = SWTImage.at<float>(row + 1, col - 1);
							if (left_down > 0 && ((*ptr) / left_down <= 5.0 || left_down / (*ptr) <= 5.0))
								boost::add_edge(this_pixel, map.at((row + 1) * SWTImage.cols + col - 1), g);
						}
					}
				}
				ptr++;
			}
		}

		std::vector<int> c(num_vertices);//得到连通域的标号    

		//以图的深度遍历得出总共有多少个连通域
		int num_comp = connected_components(g, &c[0]);

		std::vector<std::vector<SWTPoint2d> > components;
		components.reserve(num_comp);//分配vector的capacity
		//输出在过滤前的连通域和SWT点的个数
		std::cout << "Before filtering, " << num_comp << " components and " << num_vertices << "有效的笔划像素点 vertices" << std::endl;
		for (int j = 0; j < num_comp; j++) {
			std::vector<SWTPoint2d> tmp;
			components.push_back(tmp);
		}
		//将归属于连通域的点加入vector
		for (int j = 0; j < num_vertices; j++) {
			SWTPoint2d p = revmap[j];
			(components[c[j]]).push_back(p);
		}

		return components;
	}


	/*——————————————————————————————寻找有效连通域——————————————————————————————*/

	std::vector< std::vector<SWTPoint2d> >
		findLegallyConnectedComponentsRAY(Mat& SWTImage,
		std::vector<Ray> & rays)
	{
		boost::unordered_map<int, int> map;
		boost::unordered_map<int, SWTPoint2d> revmap;

		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
		int num_vertices = 0;
		// Number vertices for graph.  Associate each point with number
		//SWT点编号
		for (int row = 0; row < SWTImage.rows; row++){
			float * ptr = (float*)SWTImage.ptr(row);
			for (int col = 0; col < SWTImage.cols; col++){
				if (*ptr > 0) {
					map[row * SWTImage.cols + col] = num_vertices;
					SWTPoint2d p;
					p.x = col;
					p.y = row;
					revmap[num_vertices] = p;
					num_vertices++;
				}
				ptr++;
			}
		}

		Graph g(num_vertices);

		// Traverse and add edges to graph
		for (std::vector<Ray>::const_iterator it = rays.begin(); it != rays.end(); it++) {
			float lastSW = 0;
			int lastRow = 0;
			int lastCol = 0;
			for (std::vector<SWTPoint2d>::const_iterator it2 = it->points.begin(); it2 != it->points.end(); it2++) {
				float currentSW = SWTImage.at<float>(it2->y, it2->x);
				if (lastSW == 0) {}
				else if (lastSW / currentSW <= 3.0 || currentSW / lastSW <= 3.0){
					boost::add_edge(map.at(it2->y * SWTImage.cols + it2->x), map.at(lastRow * SWTImage.cols + lastCol), g);
				}
				lastSW = currentSW;
				lastRow = it2->y;
				lastCol = it2->x;
			}
			lastSW = 0;
			lastRow = 0;
			lastCol = 0;
		}

		std::vector<int> c(num_vertices);

		int num_comp = connected_components(g, &c[0]);

		std::vector<std::vector<SWTPoint2d> > components;
		components.reserve(num_comp);
		std::cout << "Before filtering, " << num_comp << " components and " << num_vertices << " vertices" << std::endl;
		for (int j = 0; j < num_comp; j++) {
			std::vector<SWTPoint2d> tmp;
			components.push_back(tmp);
		}
		for (int j = 0; j < num_vertices; j++) {
			SWTPoint2d p = revmap[j];
			(components[c[j]]).push_back(p);
		}

		return components;
	}

	//计算某连通域的均值、方差、中值、最大值最小值
	//用于过滤连通域
	void componentStats(Mat& SWTImage,
		const std::vector<SWTPoint2d> & component,
		float & mean, float & variance, float & median,
		int & minx, int & miny, int & maxx, int & maxy)
	{
		std::vector<float> temp;
		temp.reserve(component.size());
		mean = 0;
		variance = 0;
		minx = 1000000;
		miny = 1000000;
		maxx = 0;
		maxy = 0;
		for (std::vector<SWTPoint2d>::const_iterator it = component.begin(); it != component.end(); it++) {
			//得到某点的SWT值,Mat的随机访问
			float t = SWTImage.at<float>(it->y, it->x);
			mean += t;
			temp.push_back(t);
			miny = std::min(miny, it->y);
			minx = std::min(minx, it->x);
			maxy = std::max(maxy, it->y);
			maxx = std::max(maxx, it->x);
		}
		mean = mean / ((float)component.size());
		for (std::vector<float>::const_iterator it = temp.begin(); it != temp.end(); it++) {
			variance += (*it - mean) * (*it - mean);
		}
		variance = variance / ((float)component.size());
		std::sort(temp.begin(), temp.end());
		median = temp[temp.size() / 2];
	}

	//************过滤连通域，计算量大，耗时！！！***********
	void filterComponents(Mat& SWTImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<std::vector<SWTPoint2d> > & validComponents,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians,
		std::vector<SWTPoint2d> & compDimensions,
		std::vector<SWTPointPair2d > & compBB)
	{
		//初始化数组容量
		validComponents.reserve(components.size());
		compCenters.reserve(components.size());
		compMedians.reserve(components.size());
		compDimensions.reserve(components.size());
		// bounding boxes
		compBB.reserve(components.size());
		for (std::vector<std::vector<SWTPoint2d> >::iterator it = components.begin(); it != components.end(); it++) {
			// compute the stroke width mean, variance, median
			//计算连通域笔划宽度均值、变化、中值
			float mean, variance, median;
			int minx, miny, maxx, maxy;
			componentStats(SWTImage, (*it), mean, variance, median, minx, miny, maxx, maxy);

			// check if variance is less than half the mean
			//方差大于均值的一半则舍弃，说明连通域的SWT值不统一，为字符的可能性较小


			//if (variance > 0.5 * mean) {
				//continue;
			//}

			//计算连通域的长度和宽度
			float length = (float)(maxx - minx + 1);
			float width = (float)(maxy - miny + 1);

			// check font height
			//高度小于300则舍弃
		/*	if (width > 300) {
				continue;
			}*/

			float area = length * width;
			float rminx = (float)minx;
			float rmaxx = (float)maxx;
			float rminy = (float)miny;
			float rmaxy = (float)maxy;
			// compute the rotated bounding box
			//计算bounding box
			float increment = 1. / 36.;
			for (float theta = increment * PI; theta<PI / 2.0; theta += increment * PI) {
				float xmin, xmax, ymin, ymax, xtemp, ytemp, ltemp, wtemp;
				xmin = 1000000;
				ymin = 1000000;
				xmax = 0;
				ymax = 0;
				//对连通域中的每个点进行旋转计算，取整个旋转过程中的极值
				for (unsigned int i = 0; i < (*it).size(); i++) {
					//为什么这样计算？
					xtemp = (*it)[i].x * cos(theta) + (*it)[i].y * -sin(theta);
					ytemp = (*it)[i].x * sin(theta) + (*it)[i].y * cos(theta);
					xmin = std::min(xtemp, xmin);
					xmax = std::max(xtemp, xmax);
					ymin = std::min(ytemp, ymin);
					ymax = std::max(ytemp, ymax);
				}
				ltemp = xmax - xmin + 1;
				wtemp = ymax - ymin + 1;
				if (ltemp*wtemp < area) {
					area = ltemp*wtemp;
					length = ltemp;
					width = wtemp;
				}
			}
			// check if the aspect ratio is between 1/10 and 10
			//字符的宽高比超出一定范围，则认为不是字符，滤除
			if (length / width < 1. / 10. || length / width > 10.) {
				continue;
			}

			// compute the diameter TODO finish
			// compute dense representation of component
			//连通域的直径和密度表示
			std::vector <std::vector<float> > denseRepr;
			denseRepr.reserve(maxx - minx + 1);
			for (int i = 0; i < maxx - minx + 1; i++) {
				std::vector<float> tmp;
				tmp.reserve(maxy - miny + 1);
				denseRepr.push_back(tmp);
				for (int j = 0; j < maxy - miny + 1; j++) {
					
						denseRepr[i].push_back(0);
				}
			}
			for (std::vector<SWTPoint2d>::iterator pit = it->begin(); pit != it->end(); pit++) {
				(denseRepr[pit->x - minx])[pit->y - miny] = 1;
			}
			// create graph representing components
			//创建图代表连通域
			const int num_nodes = it->size();
			//中间点坐标
			Point2dFloat center;
			center.x = ((float)(maxx + minx)) / 2.0;
			center.y = ((float)(maxy + miny)) / 2.0;

			//长宽值，旋转过
			SWTPoint2d dimensions;
			dimensions.x = maxx - minx + 1;
			dimensions.y = maxy - miny + 1;

			//最小值点
			SWTPoint2d bb1;
			bb1.x = minx;
			bb1.y = miny;

			//最大值点
			SWTPoint2d bb2;
			bb2.x = maxx;
			bb2.y = maxy;
			SWTPointPair2d pair(bb1, bb2);

			compBB.push_back(pair);
			compDimensions.push_back(dimensions);
			compMedians.push_back(median);
			compCenters.push_back(center);
			validComponents.push_back(*it);
		}
		std::vector<std::vector<SWTPoint2d > > tempComp;
		std::vector<SWTPoint2d > tempDim;
		std::vector<float > tempMed;
		std::vector<Point2dFloat > tempCenters;
		std::vector<SWTPointPair2d > tempBB;
		tempComp.reserve(validComponents.size());
		tempCenters.reserve(validComponents.size());
		tempDim.reserve(validComponents.size());
		tempMed.reserve(validComponents.size());
		tempBB.reserve(validComponents.size());
		//对每个连通域，计算其包含的连通域的数量，若其包含的连通域的数量大于2，则将该连通域舍弃
		for (unsigned int i = 0; i < validComponents.size(); i++) {
			int count = 0;
			for (unsigned int j = 0; j < validComponents.size(); j++) {
				if (i != j) {
					if (compBB[i].first.x <= compCenters[j].x && compBB[i].second.x >= compCenters[j].x &&
						compBB[i].first.y <= compCenters[j].y && compBB[i].second.y >= compCenters[j].y) {
						count++;
					}
				}
			}
			if (count < 2) {
				tempComp.push_back(validComponents[i]);
				tempCenters.push_back(compCenters[i]);
				tempMed.push_back(compMedians[i]);
				tempDim.push_back(compDimensions[i]);
				tempBB.push_back(compBB[i]);
			}
		}
		//更新有效区域
		validComponents = tempComp;
		compDimensions = tempDim;
		compMedians = tempMed;
		compCenters = tempCenters;
		compBB = tempBB;

		//改变容量大小到元素个数
		compDimensions.reserve(tempComp.size());
		compMedians.reserve(tempComp.size());
		compCenters.reserve(tempComp.size());
		validComponents.reserve(tempComp.size());
		compBB.reserve(tempComp.size());

		std::cout << "After filtering " << validComponents.size() << " components" << std::endl;
	}
	//************过滤连通域，计算量大，耗时！！！***********





	//两字符对有一端是相同的
	bool sharesOneEnd(Chain c0, Chain c1) {
		if (c0.p == c1.p || c0.p == c1.q || c0.q == c1.q || c0.q == c1.p) {
			return true;
		}
		else {
			return false;
		}
	}

	//距离从小到大排
	bool chainSortDist(const Chain &lhs, const Chain &rhs) {
		return lhs.dist < rhs.dist;
	}

	bool chainSortLength(const Chain &lhs, const Chain &rhs) {
		return lhs.components.size() > rhs.components.size();
	}

	std::vector<Chain> makeChains(const Mat& colorImage,
		std::vector<std::vector<SWTPoint2d> > & components,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians,
		std::vector<SWTPoint2d> & compDimensions,
		std::vector<SWTPointPair2d > & compBB) {
		assert(compCenters.size() == components.size());
		// make vector of color averages
		//每个连通域的颜色均值
		std::vector<Point3dFloat> colorAverages;
		colorAverages.reserve(components.size());
		for (std::vector<std::vector<SWTPoint2d> >::iterator it = components.begin(); it != components.end(); it++) {
			Point3dFloat mean;
			//Mat中若是三通道的值，按列排在一起
			mean.x = 0;
			mean.y = 0;
			mean.z = 0;
			int num_points = 0;
			for (std::vector<SWTPoint2d>::iterator pit = it->begin(); pit != it->end(); pit++) {
				mean.x += (float)colorImage.at<uchar>(pit->y, (pit->x) * 3);
				mean.y += (float)colorImage.at<uchar>(pit->y, (pit->x) * 3 + 1);
				mean.z += (float)colorImage.at<uchar>(pit->y, (pit->x) * 3 + 2);
				num_points++;
			}
			//计算一个连通域中所有点的平均三原色值
			mean.x = mean.x / ((float)num_points);
			mean.y = mean.y / ((float)num_points);
			mean.z = mean.z / ((float)num_points);
			colorAverages.push_back(mean);
		}

		// form all eligible pairs and calculate the direction of each
		//组成合适的字符对，并计算字符对的方向
		std::vector<Chain> chains;
		for (unsigned int i = 0; i < components.size(); i++) {
			for (unsigned int j = i + 1; j < components.size(); j++) {
				// TODO add color metric
				//两字符连通域的SWT中值相差不大，高度相差不大，则计算两字符中间点的距离，颜色距离，小于一定值则组成文本行的字符对
				if ((compMedians[i] / compMedians[j] <= 2.0 || compMedians[j] / compMedians[i] <= 2.0) &&
					(compDimensions[i].y / compDimensions[j].y <= 2.0 || compDimensions[j].y / compDimensions[i].y <= 2.0)) {
					float dist = sqrt((compCenters[i].x - compCenters[j].x) * (compCenters[i].x - compCenters[j].x) +
						(compCenters[i].y - compCenters[j].y) * (compCenters[i].y - compCenters[j].y));
					float colorDist = (colorAverages[i].x - colorAverages[j].x) * (colorAverages[i].x - colorAverages[j].x) +
						(colorAverages[i].y - colorAverages[j].y) * (colorAverages[i].y - colorAverages[j].y) +
						(colorAverages[i].z - colorAverages[j].z) * (colorAverages[i].z - colorAverages[j].z);
					if (dist < 3 * (float)(std::max(std::min(compDimensions[i].x, compDimensions[i].y), std::min(compDimensions[j].x, compDimensions[j].y)))
						&& colorDist < 1600) {
						Chain c;
						c.p = i;
						c.q = j;
						std::vector<int> comps;
						comps.push_back(c.p);
						comps.push_back(c.q);
						c.components = comps;
						c.dist = dist;
						//计算方向和距离
						float d_x = (compCenters[i].x - compCenters[j].x);
						float d_y = (compCenters[i].y - compCenters[j].y);
						float mag = sqrt(d_x*d_x + d_y*d_y);
						d_x = d_x / mag;
						d_y = d_y / mag;
						Point2dFloat dir;
						dir.x = d_x;
						dir.y = d_y;
						c.direction = dir;
						chains.push_back(c);
					}
				}
			}
		}
		std::cout << chains.size() << " eligible pairs" << std::endl;
		std::sort(chains.begin(), chains.end(), &chainSortDist);
		int k = 0;
		for (vector<Chain> ::iterator it = chains.begin(); it != chains.end(); it++, k++){
			cout << "第" << k << "组的结果 " << it->p << " " << it->q << endl;
		}
			

		std::cerr << std::endl;
		const float strictness = PI / 6.0;
		//字符对的融合
		int merges = 1;
		while (merges > 0) {
			//所有对初始化为未融合
			for (unsigned int i = 0; i < chains.size(); i++) {
				chains[i].merged = false;
			}
			merges = 0;
			std::vector<Chain> newchains;
			for (unsigned int i = 0; i < chains.size(); i++) {
				for (unsigned int j = 0; j < chains.size(); j++) {
					if (i != j) {
						//未融合过且有一头相同
						if (!chains[i].merged && !chains[j].merged && sharesOneEnd(chains[i], chains[j])) {

							//左边重合，方向是否在范围内？在的话则
							if (chains[i].p == chains[j].p) {
								
							if (acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
								cout << endl;
								cout << "情况1 ： 左边重合： " << endl;
								cout << "chains[" << i << "].p = " << chains[i].p << " chains[" << i << "].q = " << chains[i].q;
								cout << "chains[" << j << "].p = " << chains[j].p << "chains[" << j << "].q = " << chains[j].q;
								cout << "i = " << i << " j = " << j << "  chains[i].direction.x = " << chains[i].direction.x <<
									"  chains[j].direction.x = "
									<< chains[j].direction.x << endl;
								cout << "值 = " << chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y << endl;
								cout << "角度 = " << acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) << endl;

									chains[i].p = chains[j].q;
									for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
										chains[i].components.push_back(*it);
									}
									float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
									float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
									chains[i].dist = d_x * d_x + d_y * d_y;

									float mag = sqrt(d_x*d_x + d_y*d_y);
									d_x = d_x / mag;
									d_y = d_y / mag;
									Point2dFloat dir;
									dir.x = d_x;
									dir.y = d_y;
									chains[i].direction = dir;
									chains[j].merged = true;
									merges++;
									
								}
							}
							else if (chains[i].p == chains[j].q) {
								if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
									cout << endl;
									cout << "情况2 ： 左与右重合： " << endl;
									cout << "chains[" << i << "].p = " << chains[i].p << " chains[" << i << "].q = " << chains[i].q;
									cout << "chains[" << j << "].p = " << chains[j].p << "chains[" << j << "].q = " << chains[j].q;
									cout << "i = " << i << " j = " << j << "  chains[i].direction.x = " << chains[i].direction.x <<
										"  chains[j].direction.x = "
										<< chains[j].direction.x << endl;
									cout << "值 = " << chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y << endl;
									cout << "角度 = " << acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) << endl;
									chains[i].p = chains[j].p;
									for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
										chains[i].components.push_back(*it);
									}
									float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
									float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
									float mag = sqrt(d_x*d_x + d_y*d_y);
									chains[i].dist = d_x * d_x + d_y * d_y;

									d_x = d_x / mag;
									d_y = d_y / mag;

									Point2dFloat dir;
									dir.x = d_x;
									dir.y = d_y;
									chains[i].direction = dir;
									chains[j].merged = true;
									merges++;
									}
							}
							else if (chains[i].q == chains[j].p) {
								if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
									cout << endl;
									cout << "情况3 ： 右边与左边重合： " << endl;
									cout << "chains[" << i << "].p = " << chains[i].p << " chains[" << i << "].q = " << chains[i].q;
									cout << "chains[" << j << "].p = " << chains[j].p << "chains[" << j << "].q = " << chains[j].q;
									cout << "i = " << i << " j = " << j << "  chains[i].direction.x = " << chains[i].direction.x <<
										"  chains[j].direction.x = "
										<< chains[j].direction.x << endl;
									cout << "值 = " << chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y << endl;
									cout << "角度 = " << acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) << endl;
									//更新第一对的尾，更新第一组的相连部分，更新相连chain的距离和角度
									chains[i].q = chains[j].q;
									for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
										chains[i].components.push_back(*it);
									}
									float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
									float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
									float mag = sqrt(d_x*d_x + d_y*d_y);
									chains[i].dist = d_x * d_x + d_y * d_y;


									d_x = d_x / mag;
									d_y = d_y / mag;
									Point2dFloat dir;
									dir.x = d_x;
									dir.y = d_y;

									chains[i].direction = dir;
									chains[j].merged = true;
									merges++;
									}
							}
							else if (chains[i].q == chains[j].q) {
								if (acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
									cout << endl;
									cout << "情况4：右边和右边重合 " << endl;
									cout << "chains[" << i << "].p = " << chains[i].p << " chains[" << i << "].q = " << chains[i].q;
									cout << "chains[" << j << "].p = " << chains[j].p << "chains[" << j << "].q = " << chains[j].q;
									cout << "i = " << i << " j = " << j << "  chains[i].direction.x = " << chains[i].direction.x <<
										"  chains[j].direction.x = "
										<< chains[j].direction.x << endl;
									cout << "值 = " << chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y << endl;
									cout << "角度 = " << acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) << endl;
									chains[i].q = chains[j].p;
									for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
										chains[i].components.push_back(*it);
									}
									float d_x = (compCenters[chains[i].p].x - compCenters[chains[i].q].x);
									float d_y = (compCenters[chains[i].p].y - compCenters[chains[i].q].y);
									chains[i].dist = d_x * d_x + d_y * d_y;

									float mag = sqrt(d_x*d_x + d_y*d_y);
									d_x = d_x / mag;
									d_y = d_y / mag;
									Point2dFloat dir;
									dir.x = d_x;
									dir.y = d_y;
									chains[i].direction = dir;
									chains[j].merged = true;
									merges++;
									}
							}
						}
					}
				}
			}
			for (unsigned int i = 0; i < chains.size(); i++) {
				if (!chains[i].merged) {
					newchains.push_back(chains[i]);
				}
			}
			chains = newchains;
			std::stable_sort(chains.begin(), chains.end(), &chainSortLength);
		}

		std::vector<Chain> newchains;
		newchains.reserve(chains.size());
		for (std::vector<Chain>::iterator cit = chains.begin(); cit != chains.end(); cit++) {
			if (cit->components.size() >= 3) {
				newchains.push_back(*cit);
			}
		}
		chains = newchains;
		std::cout << chains.size() << " chains after merging" << std::endl;
		return chains;
	}

}

