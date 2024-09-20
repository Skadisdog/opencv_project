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
    cv::Mat mask; // 掩码，就是一张黑白图像
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
            cv::rectangle(imageWithBoundingBox, boundingBox, cv::Scalar(0, 0, 255), 1); // 红色边界框，线宽1
            // 画轮廓
            cv::drawContours(contourImage, contours, (int)i, cv::Scalar(0, 0, 255), 1.2);

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
    //灰度图使用最开始创建的那个
    cv::Mat binaryImage;
    double thresh = 200; // 阈值，可调
    double maxValue = 255; // 二值化后最大值
    cv::threshold(grayImage, binaryImage, thresh, maxValue, cv::THRESH_BINARY);
    output(binaryImage, "..//output//binary.png");
    // 创建卷积核
    int morph_size = 2; // 核大小
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, // 核的形状，这里采用矩形
                                                cv::Size(2 * morph_size + 1, 2 * morph_size + 1) // 大小
                                                );

    // 第一次腐蚀，消除小白点噪声
    cv::Mat firstEroded;
    cv::erode(binaryImage, firstEroded, element);

    // 第一次膨胀，将图像膨胀到原来大小
    cv::Mat firstDilated;
    cv::dilate(firstEroded, firstDilated, element);

    // 第二次膨胀，去除黑色噪点
    cv::Mat secondDilated;
    cv::dilate(firstDilated, secondDilated, element);

    // 第二次腐蚀，腐蚀到原来大小
    cv::Mat secondEroded;
    cv::erode(secondDilated, secondEroded, element);

    output(firstDilated, "..//output//dilated.png");
    output(secondEroded, "..//output//eroded.png");

    // 设置种子点
    cv::Point seedPoint(0, 0); // 根据需要调整种子点的位置

    // 设置填充颜色（对于二值图像，填充颜色为0或255）
    int newColor = 128; // 中间灰色

    // 定义填充的阈值范围（对于二值图像，通常设置为0）
    int loDiff = 0;
    int upDiff = 0;

    // 漫水填充
    cv::Mat dst = secondEroded.clone();
    cv::floodFill(  
                    dst, 
                    seedPoint, //起始点
                    cv::Scalar(newColor), // 填充颜色
                    0, // 是否存储边界矩形
                    cv::Scalar(loDiff), 
                    cv::Scalar(upDiff)
                );

    output(dst, "..//output//flood fill.png");

    // 文字在之前已经绘制完成，现在进行方圆的绘制
    cv::Mat paintingImage = cv::Mat::zeros(800, 800, CV_8UC3);
    // 设置圆心和半径
    cv::Point center(400, 400); // 圆心坐标
    int radius = 50; // 半径
    // 绘制圆
    cv::circle(paintingImage, center, radius, cv::Scalar(0, 0, 255), 2); // 2 表示线条粗细

    // 设置矩形的左上角和右下角坐标
    cv::Point topLeft(200, 200);
    cv::Point bottomRight(600, 600);
    // 绘制矩形
    cv::rectangle(paintingImage, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2); // 2 表示线条粗细
    output(paintingImage, "..//output//painting.png");

    // 红色的外轮廓绘制了，此处省略

    // 获取图像中心点
    cv::Point2f centerToRatate(image.cols / 2.0, image.rows / 2.0);

    // 设置旋转角度
    double angle = 35.0; // 旋转角度，可以根据需要调整

    // 设置缩放比例
    double scale = 1.0; // 缩放比例，可以根据需要调整

    // 计算旋转矩阵
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(centerToRatate, angle, scale);

    // 计算旋转后图像的边界，为了防止丢失信息
    cv::Rect bbox = cv::RotatedRect(centerToRatate, image.size(), angle).boundingRect();

    // 调整旋转矩阵中的平移部分
    rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - centerToRatate.x;
    rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - centerToRatate.y;

    // 应用旋转矩阵
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotationMatrix, bbox.size());

    output(rotatedImage, "..//output//rotated.png");

    // 获取图像的宽度和高度
    int width = image.cols;
    int height = image.rows;

    // 定义裁剪区域（左上角 1/4 区域）
    cv::Rect roi(0, 0, width / 2, height / 2);

    // 裁剪图像
    cv::Mat croppedImage = image(roi);

    output(croppedImage, "..//output//cropped.png");

    return 0;
}
int output(cv::Mat image, string outputPath){
    if (!cv::imwrite(outputPath, image)) {
        cerr << "Failed to save the image:" << outputPath << endl;
    }
    return 0;
}