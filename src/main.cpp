#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;

// 输出函数
int output(cv::Mat image, string outputPath);

int main(){
    // 读取图像
    cv::Mat image = cv::imread("..//resources//test_image.png", cv::IMREAD_COLOR);
    output(image, "origin.png");
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // 转换为灰度图像
    output(grayImage, "grey.png");
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV); // 转换为hsv
    output(hsvImage, "hsv.png");
    cv::Mat blurImage;
    cv::blur(image, blurImage, cv::Size(10, 10)); // 均值滤波 内核10X10
    output(blurImage, "blur.png");
    cv::Mat gaussianBlurredImage;
    cv::GaussianBlur(image, gaussianBlurredImage, cv::Size(9, 9), 1.5); // 高斯滤波
    output(gaussianBlurredImage, "Gaussianblur.png");
    cv::Mat redBGRImage;
    cv::Mat mask; // 掩码，就是一张黑白图像，像是粥解包出来的通道图一样
    cv::Scalar lowerRed(0, 0, 100); // 红色的下限
    cv::Scalar upperRed(120, 120, 255); // 红色的上限
    // 创建掩码
    cv::inRange(image, lowerRed, upperRed, mask);
    // 提取红色区域
    cv::bitwise_and(image, image, redBGRImage, mask);
    output(redBGRImage, "red BGR.png");
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
    output(redHSVImage, "red HSV.png");
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // 找到外轮廓
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 绘制外轮廓到一个新的图像上
    cv::Mat contourImage = cv::Mat::zeros(mask.size(), CV_8UC3);
    cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2); // 绿色轮廓，线宽2
    return 0;
}
int output(cv::Mat image, string outputPath){
    if (cv::imwrite(outputPath, image)) {
        cout << "Image saved successfully to " << outputPath << endl;
    } else {
        cerr << "Failed to save the image" << endl;
    }
    return 0;
}