#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;

// 输出函数
int output(cv::Mat image, string outputPath);

int main(){
    // 读取图像
    cv::Mat image = cv::imread("..//resources//test_image.png", cv::IMREAD_COLOR);
    output(image, "..//output//origin.png");
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // 转换为灰度图像
    output(grayImage, "..//output//grey.png");
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV); // 转换为hsv
    output(hsvImage, "..//output//hsv.png");
    cv::Mat blurImage;
    cv::blur(image, blurImage, cv::Size(10, 10)); // 均值滤波 内核10X10
    output(blurImage, "..//output//blur.png");
    cv::Mat gaussianBlurredImage;
    cv::GaussianBlur(image, gaussianBlurredImage, cv::Size(9, 9), 1.5); // 高斯滤波
    output(gaussianBlurredImage, "..//output//Gaussianblur.png");
    cv::Mat redBGRImage;
    cv::Mat mask; // 掩码，就是一张黑白图像，像是粥解包出来的通道图一样
    cv::Scalar lowerRed(0, 0, 100); // 红色的下限
    cv::Scalar upperRed(120, 120, 255); // 红色的上限
    // 创建掩码
    cv::inRange(image, lowerRed, upperRed, mask);
    // 提取红色区域
    cv::bitwise_and(image, image, redBGRImage, mask);
    output(redBGRImage, "..//output//red BGR.png");
    // 定义红色的HSV范围
    cv::Scalar lowerRed1(0, 100, 100);
    cv::Scalar upperRed1(10, 255, 255);
    cv::Scalar lowerRed2(160, 100, 100);
    cv::Scalar upperRed2(179, 255, 255);
    // 创建掩码
    cv::Mat mask1, mask2;
    cv::inRange(hsvImage, lowerRed1, upperRed1, mask1);
    cv::inRange(hsvImage, lowerRed2, upperRed2, mask2);
    cv::bitwise_or(mask1, mask2, mask);
    // 提取红色区域
    cv::Mat redHSVImage;
    cv::bitwise_and(image, image, redHSVImage, mask);
    output(redHSVImage, "..//output//red HSV.png");
    cv::Mat maskToContours = mask.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // 找到外轮廓
    cv::findContours(maskToContours, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // RETR_EXTERNAL 只找外轮廓
    // CHAIN_APPROX_SIMPLE 压缩找的contours的点的数量
    // 原始图像会被修改，所以要复制一份
    // 绘制外轮廓到一个新的图像上
    //cv::Mat contourImage = image.clone();
    cv::Mat contourImage = cv::Mat::zeros(mask.size(), CV_8UC3);
    cv::Mat imageWithBoundingBox = image.clone();
    int contourIdx = 0;
        for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        // 过滤掉面积过小的轮廓
        if (area > 10) // 可以根据具体情况调整阈值
        {
            // 画外接矩形
            cv::Rect boundingBox = cv::boundingRect(contours[i]);
            cv::rectangle(imageWithBoundingBox, boundingBox, cv::Scalar(0, 255, 0), 1); // 绿色边界框，线宽1
            // 画轮廓
            cv::drawContours(contourImage, contours, (int)i, cv::Scalar(0, 255, 0), 1.2);

            // 创建一个掩码图像用于距离变换
            cv::Mat contourMask = cv::Mat::zeros(mask.size(), CV_8UC1);
            cv::drawContours(contourMask, contours, (int)i, cv::Scalar(255), cv::FILLED); // 填充

            // 距离变换
            cv::Mat distTransform;
            cv::distanceTransform(contourMask, distTransform, cv::DIST_L2, 5);

            // 找到距离变换图像中的最大值及其位置
            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(distTransform, nullptr, &maxVal, nullptr, &maxLoc);

            // 在轮廓上编号
            cv::putText(contourImage, std::to_string(contourIdx), maxLoc, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 0.5);

            // 输出编号和对应的轮廓面积
            cout << "Contour #" << contourIdx << " area: " << area << endl;

            contourIdx++;
        }
    }
    output(contourImage, "..//output//contour.png");
    output(imageWithBoundingBox, "..//output//bounding box.png");
    return 0;
}
int output(cv::Mat image, string outputPath){
    if (!cv::imwrite(outputPath, image)) {
        cerr << "Failed to save the image:" << outputPath << endl;
    }
    return 0;
}