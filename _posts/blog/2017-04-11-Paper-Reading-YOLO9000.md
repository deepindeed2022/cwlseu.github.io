---
layout: post
title: "论文笔记：YOLO9000: Better,Faster,Stronger"
categories: [blog]
tags: [CV算法, detection]
description: 这一网络结构可以实时地检测超过9000种物体分类，这归功于它使用了WordTree，通过WordTree来混合检测数据集与识别数据集之中的数据。
---

{:toc}

## 摘要
1. 提出YOLO v2 ：代表着目前业界最先进物体检测的水平，它的速度要快过其他检测系统（FasterR-CNN，ResNet，SSD），使用者可以在它的速度与精确度之间进行权衡。
2. 提出了一种新的联合训练算法（ Joint Training Algorithm ），使用这种联合训练技术同时在ImageNet和COCO数据集上进行训练。YOLO9000进一步缩小了监测数据集与识别数据集之间的代沟。这种算法可以把这两种的数据集混合到一起。使用一种分层的观点对物体进行分类，用巨量的分类数据集数据来扩充检测数据集，从而把两种不同的数据集混合起来。 
联合训练算法的基本思路就是：同时在检测数据集(COCO)和分类数据集(ImageNet)上训练物体检测器（Object Detectors)，用检测数据集的数据学习物体的准确位置，用分类数据集的数据来增加分类的类别量、提升健壮性。 
3. 提出YOLO9000 ：这一网络结构可以实时地检测超过9000种物体分类，这归功于它使用了WordTree，通过WordTree来混合检测数据集与识别数据集之中的数据。
All of our code and pre-trained models are available online at <http://pjreddie.com/yolo9000/>.
说白了一点，就是YOLOv2是在一个混合的大的数据集合上进行训练，然后在VOC2007数据集合上进行测试结果的。相比较之前的SSD等方法感觉有点不太公平。不过YOLO作者生撸硬调还是值得膜拜的。

## BETTER
YOLO一代有很多缺点，作者希望改进的方向是:改善recall，提升定位的准确度，同时保持分类的准确度。YOLO2主要木雕是为了简化网络。 
![@](https://cwlseu.github.io/images/yolo/模型设计考虑因素.png)

### Batch Normalization
使用按批归一化对网络进行优化，让网络提高了收敛性，同时还消除了对其他形式的正则化（regularization）的依赖。通过对YOLO的每一个卷积层增加按批归一化， 最终使得mAP提高了2%，同时还使model正则化。 使用Batch Normalization可以从model中去掉Dropout，而不会产生过拟合。

### High resolution classifier
目前业界标准的检测方法，都要先把分类器（classiﬁer）放在ImageNet上进行预训练。从Alexnet开始，大多数的分类器都运行在小于256x256的图片上。而现在YOLO从224x224增加到了448x448，这就意味着网络需要适应新的输入分辨率。 
为了适应新的分辨率，YOLO v2的分类网络以448x448的分辨率先在ImageNet上进行Fine Tune，Fine Tune10个epochs，给一定的时间让网络调整他的滤波器（filters），好让其能更好的运行在新分辨率上，还需要调优用于检测的Resulting Network。最终通过使用高分辨率，mAP提升了4%。

### Convolution with anchor boxes

YOLO一代包含有全连接层，从而能直接预测Bounding Boxes的坐标值。 Faster R-CNN的方法只用卷积层与Region Proposal Network来预测Anchor Box的偏移值与置信度，而不是直接预测坐标值。作者发现通过预测偏移量而不是坐标值能够简化问题，让神经网络学习起来更容易。所以最终YOLO去掉了全连接层，使用Anchor Boxes来预测 Bounding Boxes。作者去掉了网络中一个Pooling层，这让卷积层的输出能有更高的分辨率。收缩网络让其运行在416*416而不是448*448。由于图片中的物体都倾向于出现在图片的中心位置，特别是那种比较大的物体，所以有一个单独位于物体中心的位置用于预测这些物体。YOLO的卷积层采用32这个值来下采样图片，所以通过选择416*416用作输入尺寸最终能输出一个13*13的Feature Map。 使用Anchor Box会让精确度稍微下降，但用了它能让YOLO能预测出大于**一千个框**，同时recall达到88%，mAP达到69.2%。

### Dimension clusters

**之前Anchor Box的尺寸是手动选择的，所以尺寸还有优化的余地。** 为了优化，在训练集（training set）Bounding Boxes上跑了一下k-means聚类，来找到一个比较好的值。 
如果我们用标准的欧式距离的k-means，**尺寸大的框比小框产生更多的错误**。因为我们的目的是提高先验框的IOU分数，这不能依赖于Box的大小，所以距离度量的使用：

	d(box, centroid) = 1 - IOU(box, centroid)


这个地方我就特别想了想如何实现，就扒OpenCV来看看。



```cpp
static inline float normL2Sqr(const float* a, const float* b, int n)
{
    float s = 0.f;
    for( int i = 0; i < n; i++ )
    {
        float v = a[i] - b[i];
        s += v*v;
    }
    return s;
}

class KMeansDistanceComputer : public ParallelLoopBody
{
public:
    KMeansDistanceComputer( double *_distances,
                            int *_labels,
                            const Mat& _data,
                            const Mat& _centers )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers)
    {
    }

    void operator()( const Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        for( int i = begin; i<end; ++i)
        {
            const float *sample = data.ptr<float>(i);
            int k_best = 0;
            double min_dist = DBL_MAX;

            for( int k = 0; k < K; k++ )
            {
                const float* center = centers.ptr<float>(k);
                // 调用欧拉距离计算函数，具体在上面函数中实现
                const double dist = normL2Sqr(sample, center, dims);

                if( min_dist > dist )
                {
                    min_dist = dist;
                    k_best = k;
                }
            }

            distances[i] = min_dist;
            labels[i] = k_best;
        }
    }

private:
    KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const Mat& data;
    const Mat& centers;
};

// 
// @data –
// Data for clustering. An array of N-Dimensional points with float coordinates is needed. Examples of this array can be:

// Mat points(count, 2, CV_32F);
// Mat points(count, 1, CV_32FC2);
// Mat points(1, count, CV_32FC2);
// std::vector<cv::Point2f> points(sampleCount);
// @cluster_count – Number of clusters to split the set by.
// @K – Number of clusters to split the set by.
// @_best_labels – Input/output integer array that stores the cluster indices for every sample.
// @criteria – The algorithm termination criteria, that is, the maximum number of iterations and/or the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
// @attempts – Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness (see the last function parameter).
// @flags –
// Flag that can take the following values:

// 		KMEANS_RANDOM_CENTERS Select random initial centers in each attempt.
// 		KMEANS_PP_CENTERS Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
// 		KMEANS_USE_INITIAL_LABELS During the first (and possibly the only) attempt, use the user-supplied labels instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random centers. Use one of KMEANS_*_CENTERS flag to specify the exact method.
//@_centers – Output matrix of the cluster centers, one row per each cluster center.


double cv::kmeans( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers )
{
    const int SPP_TRIALS = 3;
    Mat data0 = _data.getMat();
    bool isrow = data0.rows == 1;
    int N = isrow ? data0.cols : data0.rows;
    int dims = (isrow ? 1 : data0.cols)*data0.channels();
    int type = data0.depth();

    attempts = std::max(attempts, 1);
    CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

    _bestLabels.create(N, 1, CV_32S, -1, true);

    Mat _labels, best_labels = _bestLabels.getMat();
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    std::vector<int> counters(K);
    std::vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];
    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;
    // 结束阈值
    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }
    // sample: Floating-point matrix of input samples, one row per sample.
    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0;; )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                // 初始化中心位置
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    // 随机生成中心
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    j=0;
                    #if CV_ENABLE_UNROLLED
                    for(; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    #endif
                    for( ; j < dims; j++ )
                        center[j] += sample[j];
                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;
                // 处理聚类中心个数为0的情况
                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0;
                    for( int k1 = 1; k1 < K; k1++ )
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center = centers.ptr<float>(k);
                    float* old_center = centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];
                    for( j = 0; j < dims; j++ )
                        _old_center[j] = old_center[j]*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);

                        // 距离采用传统的欧氏距离
                        double dist = normL2Sqr(sample, _old_center, dims);

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    for( j = 0; j < dims; j++ )
                    {
                        old_center[j] -= sample[j];
                        new_center[j] += sample[j];
                    }
                }
                // 计算新旧中心点的偏差和
                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/counters[k];
                    for( j = 0; j < dims; j++ )
                        center[j] *= scale;

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }
            // 迭代次数达到一定次数，结束；中心变化小于一定阈值结束
            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            Mat dists(1, N, CV_64F);
            double* dist = dists.ptr<double>(0);
            // 这里计算center和sample之间的距离，默认采用欧氏距离
            // 这里也是需要修改的KMeansDistanceComputer(dist, labels, data, centers)
            // 距离的计算方法
            parallel_for_(Range(0, N),
                         KMeansDistanceComputer(dist, labels, data, centers));
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                compactness += dist[i];
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
                centers.copyTo(_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}

```

iou函数在opencv中有computeOneToOneMatchedOverlaps的实现，当然我们也可以自己写一个。

```cpp
// 自己实现一把IOU计算
static inline float intersection_over_union(const float* boxA, const float* boxB)
{
	//determine the (x, y)-coordinates of the intersection rectangle
    float xA = max(boxA[0], boxB[0]);
    float yA = max(boxA[1], boxB[1]);
    float xB = min(boxA[2], boxB[2]);
    float yB = min(boxA[3], boxB[3]);

    // compute the area of intersection rectangle
    float interArea = (xB - xA + 1) * (yB - yA + 1);

    // compute the area of both the prediction and ground-truth
    // rectangles
    float boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
    float boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

    // compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea);
}
static inline float priorsbox_iou_dist(const float* a, const float* b, int n)
{
	return 1.0f - intersection_over_union(a, b);
}
```

<!-- <figure class="half">
    <img src="https://cwlseu.github.io/images/yolo/ClusterBox.png"  width = "280">
    <img src="https://cwlseu.github.io/images/yolo/AvgIOU.png"  width = "280">
</figure> -->
![@簇集的Box位置信息](https://cwlseu.github.io/images/yolo/ClusterBox.png)
![@不同方法进行簇集的AVGIOU结果](https://cwlseu.github.io/images/yolo/AvgIOU.png)

通过分析实验结果（Figure 2），左图：在model复杂性与high recall之间权衡之后，选择聚类分类数K=5。右图：是聚类的中心，大多数是高瘦的Box。 
Table1是说明用K-means选择Anchor Boxes时，当Cluster IOU选择值为5时，AVG IOU的值是61，这个值要比不用聚类的方法的60.9要高。选择值为9的时候，AVG IOU更有显著提高。总之就是说明用聚类的方法是有效果的。

### Direct location prediction

用Anchor Box的方法，会让model变得不稳定，尤其是在最开始的几次迭代的时候（？？）。大多数不稳定因素产生自预测Box的（x,y）位置的时候。按照之前YOLO的方法，网络不会预测偏移量，而是根据YOLO中的网格单元的位置来预测坐标，这就让Ground Truth的值介于0到1之间。(这个地方在文章中感觉公式是有问题的，根本说不通啊。check之前Faster R-CNN关于anchor box中的说明)而为了让网络的结果能落在这一范围内，网络使用一个 Logistic Activation来对于网络预测结果进行限制，让结果介于0到1之间。 网络在每一个网格单元中预测出5个Bounding Boxes，每个Bounding Boxes有五个坐标值tx，ty，tw，th，t0，他们的关系见下图（Figure3）。假设一个网格单元对于图片左上角的偏移量是cx，cy，Bounding Boxes Prior的宽度和高度是pw，ph，那么预测的结果见下图右面的公式： 

![@Prediction Location and BoundingBox之间的位置关系](https://cwlseu.github.io/images/yolo/bbox-loc-prediction.png)
因为使用了限制让数值变得参数化，也让网络更容易学习、更稳定。 
Dimension clusters和Direct location prediction，improves YOLO by almost 5% over the version with anchor boxes.

### Fine-Grained Features

YOLO修改后的Feature Map大小为13*13，这个尺寸对检测图片中尺寸大物体来说足够了，同时使用这种细粒度的特征对定位小物体的位置可能也有好处。Faster R-CNN、SSD都使用不同尺寸的Feature Map来取得不同范围的分辨率，而YOLO采取了不同的方法，YOLO加上了一个Passthrough Layer来取得之前的某个26*26分辨率的层的特征。这个Passthrough layer能够把高分辨率特征与低分辨率特征联系在一起，联系起来的方法是把相邻的特征堆积在不同的Channel之中，这一方法类似与Resnet的Identity Mapping，从而把26*26*512变成13*13*2048。YOLO中的检测器位于扩展后（expanded ）的Feature Map的上方，所以他能取得细粒度的特征信息，这提升了YOLO 1%的性能。

### Multi-ScaleTraining

作者希望YOLO v2能健壮的运行于不同尺寸的图片之上，所以把这一想法用于训练model中。 
区别于之前的补全图片的尺寸的方法，YOLO v2每迭代几次都会改变网络参数。每10个Batch，网络会随机地选择一个新的图片尺寸，由于使用了下采样参数是32，所以不同的尺寸大小也选择为32的倍数{320，352…..608}，最小320*320，最大608*608，网络会自动改变尺寸，并继续训练的过程。
![@速度和准确率的权衡Table 3：在voc2007上的速度与精度](https://cwlseu.github.io/images/yolo/Precision-Speed.jpg) 
这一策略让网络在不同的输入尺寸上都能达到一个很好的预测效果，同一网络能在不同分辨率上进行检测。当输入图片尺寸比较小的时候跑的比较快，输入图片尺寸比较大的时候精度高，所以你可以在YOLO v2的速度和精度上进行权衡。 

![@性能的对比](https://cwlseu.github.io/images/yolo/Performance.PNG)
<!-- <figure class="half">
    <img src="https://cwlseu.github.io/images/yolo/Precision-Speed.png"  width = "285">
    <img src="https://cwlseu.github.io/images/yolo/Performance.png"  width = "285">
</figure>
 -->


<!-- ## Faster

YOLO使用的是Googlelent架构，比VGG-16快，YOLO完成一次前向过程只用8.52 billion运算，而VGG-16要30.69billion，但是YOLO精度稍低于VGG-16。
YOLO v2基于一个新的分类model，有点类似与VGG。YOLO v2使用3*3filter，每次Pooling之后都增加一倍Channels的数量。YOLO v2使用全局平均Pooling，使用Batch Normilazation来让训练更稳定，加速收敛，使model规范化。 
最终的model–Darknet19，有19个卷积层和5个maxpooling层，处理一张图片只需要5.58 billion次运算，在ImageNet上达到72.9%top-1精确度，91.2%top-5精确度。

### Training for classiﬁcation

网络训练在 ImageNet 1000类分类数据集，训练了160epochs，使用随机梯度下降，初始学习率为0.1， polynomial 
rate decay with a power of 4, weight decay of 0.0005 and momentum of 0.9 。训练期间使用标准的数据扩大方法：随机裁剪、旋转、变换颜色（hue）、变换饱和度（saturation）， 变换曝光度（exposure shifts）。 
在训练时，把整个网络在更大的448*448分辨率上Fine Turnning 10个 epoches，初始学习率设置为0.001，这种网络达到达到76.5%top-1精确度，93.3%top-5精确度。

### Training for detection

网络去掉了最后一个卷积层，而加上了三个3*3卷积层，每个卷积层有1024个Filters，每个卷积层紧接着一个1*1卷积层， with 
the number of outputs we need for detection。 
对于VOC数据，网络预测出每个网格单元预测五个Bounding Boxes，每个Bounding Boxes预测5个坐标和20类，所以一共125个Filters，增加了Passthough层来获取前面层的细粒度信息，网络训练了160 epoches，lr = 0.001，dividing it by 10 at 60 and 90 epochs，weightdecay=.0005  momentum=0.9，数据扩大方法相同，对COCO与VOC数据集的训练对策相同。 --> 

## 小结
YOLO9000在原来使用手工Anchor的基础上，使用聚类的anchor大小进行。同时使用多尺度图片进行训练，采用grad-cell技术最后生成13x13的特征空间图。
