# XJTU-RMV-Task02

提前声明，此程序基本由gpt-4o完成，作者仅进行了调试和排错。

向gpt提问的示例：

>给定原始图像，将图像裁剪为原图的左上角 1/4，怎么办？

gpt会根据这一问题，进行对应的回答，出乎我意料的是，它有时候会根据我前面提问过的内容，进行补充性回答，这无疑大大节省了考虑时间。

现在对整体程序进行分析，从完成思路，代码解释和问题反思三个方面进行。

### 完成思路

根据任务书要求，逐步完成，部分可合并内容，则放在一起执行，共16张图片，展示了各类处理的效果

### 代码解释

读取图像：使用`imread()`函数，从文件系统中读取图像。

灰度图转化：`cvtColor()`，提供参数`COLOR_BGR2GRAY`则将BGR图像转化为灰度图，通过加权平均，转化成黑白效果。

HSV图像转化：同理，使用`cvtColor()`，参数提供`COLOR_BGR2HSV`。

均值滤波：`blur()`，设置好滤波核，对每个像素点，与周围像素点做平均运算。

高斯滤波：`GaussianBlur()`，参数设置与均值滤波类似，注意滤波核大小必须是大于1的奇数，最后一个参数为标准差，给定之后，就可以通过高斯函数构造滤波核，这样更符合图像的自然逻辑----越靠近某一像素的其它像素和它关联性越强。

特征提取：这一部分放在一起说，首先是分别使用RGB和HSV方法提取红色区域，原理都是设置红色的范围，从而只取范围内的颜色，创造掩模（掩码），即一张单通道黑白图像，然后利用`bitwise_and()`函数，对两张图像进行与运算，从而只留下了红色的部分。值得注意的是，RGB色彩空间中，红色并不连续，因此不好选出红色，相比之下，HSV空间对红色的定义就简单、连续很多，这也是图像处理多使用HSV空间的原因。

接下来的步骤是寻找并绘制外轮廓以及bounding box。`findContours()`用来寻找外轮廓，参数的含义已经在代码中注释。寻找外轮廓的过程中，opencv会对原始图像进行标注和修改，因此如果不想破坏原始图像，需要额外建立一个新的Mat，并且还需注意`clone()`和`copyTo()`之间的区别，不能用错。

`boundingRect()`用来查找轮廓的外接矩形，`rectangle()`用来绘制矩形，值得注意的是，我先计算了每个轮廓的面积，筛去了面积小于10的轮廓。

这里我额外进行了距离变换，用来查找轮廓的最大内切圆，通过这个操作，我在一定程度上实现了在图中给轮廓编号，也就完成了后续在图片打字的任务。通过编号，即可实现轮廓和编号的一一对应。

通过已有灰度图，选择合适的阈值，再结合`threshold()`函数进行二值化，就可以筛出图片中亮度较高的部分，关于调整阈值，我还没想好具体在车上，光线条件不稳定的情况下怎么才能选择出足够好的阈值，不过考虑到车载摄像头的曝光调的非常低，阈值设置的范围可能比较宽泛。另外一件重要的事情是，白色的天空部分也会被认定为高亮区域，比赛场景中，也许可以通过筛选一定高度下的高亮部分来锁定灯带，或者进行特征检测。

接下来就是创建卷积核，进行腐蚀膨胀，这里我进行了两次腐蚀膨胀操作，目的是消除黑色和白色噪点，这里我无法确认这个做法是否正确，只是一个尝试。构造卷积核使用`getStructuringElement()`函数，腐蚀和膨胀分别是`erode()`和`dilate()`。

漫水处理使用`floodFill()`，设定初始种子，从该种子开始处理。

图像绘制：使用`circle()`绘制圆形，`rectangle()`绘制矩形。

图像处理：`getRotationMatrix2D()`用来计算旋转矩阵，`warpAffine()`将旋转矩阵应用到原图像。

裁切则是直接使用Rect类来定义裁剪区域，`image(roi)`（其中，roi是定义的Rect类）即可裁切。

### 问题反思

距离变换操作的耗时过长，也许还有优化空间，以及单文件编写有害查找错误和编译，可以优化成多文件格式，把不同功能拆分开来。