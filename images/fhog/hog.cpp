#include "cascadedetect.hpp"
#include "opencv2/core/core_c.h"
#include "opencl_kernels_objdetect.hpp"

#include <cstdio>
#include <iterator>
#include <limits>

/****************************************************************************************\
      The code below is implementation of HOG (Histogram-of-Oriented Gradients)
      descriptor and object detection, introduced by Navneet Dalal and Bill Triggs.

      The computed feature vectors are compatible with the
      INRIA Object Detection and Localization Toolkit
      (http://pascal.inrialpes.fr/soft/olt/)
\****************************************************************************************/

namespace cv
{

#define NTHREADS 256

enum {DESCR_FORMAT_COL_BY_COL, DESCR_FORMAT_ROW_BY_ROW};

static int numPartsWithin(int size, int part_size, int stride)
{
    return (size - part_size + stride) / stride;
}

static Size numPartsWithin(cv::Size size, cv::Size part_size,
                                                cv::Size stride)
{
    return Size(numPartsWithin(size.width, part_size.width, stride.width),
        numPartsWithin(size.height, part_size.height, stride.height));
}

static size_t getBlockHistogramSize(Size block_size, Size cell_size, int nbins)
{
    Size cells_per_block = Size(block_size.width / cell_size.width,
        block_size.height / cell_size.height);
    return (size_t)(nbins * cells_per_block.area());
}

size_t HOGDescriptor::getDescriptorSize() const
{
    CV_Assert(blockSize.width % cellSize.width == 0 &&
        blockSize.height % cellSize.height == 0);
    CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
        (winSize.height - blockSize.height) % blockStride.height == 0 );

    return (size_t)nbins*
        (blockSize.width/cellSize.width)*
        (blockSize.height/cellSize.height)*
        ((winSize.width - blockSize.width)/blockStride.width + 1)*
        ((winSize.height - blockSize.height)/blockStride.height + 1);
}

double HOGDescriptor::getWinSigma() const
{
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}

bool HOGDescriptor::checkDetectorSize() const
{
    size_t detectorSize = svmDetector.size(), descriptorSize = getDescriptorSize();
    return detectorSize == 0 ||
        detectorSize == descriptorSize ||
        detectorSize == descriptorSize + 1;
}

void HOGDescriptor::setSVMDetector(InputArray _svmDetector)
{
    _svmDetector.getMat().convertTo(svmDetector, CV_32F);
    CV_Assert(checkDetectorSize());

    Mat detector_reordered(1, (int)svmDetector.size(), CV_32FC1);

    size_t block_hist_size = getBlockHistogramSize(blockSize, cellSize, nbins);
    cv::Size blocks_per_img = numPartsWithin(winSize, blockSize, blockStride);

    for (int i = 0; i < blocks_per_img.height; ++i)
        for (int j = 0; j < blocks_per_img.width; ++j)
        {
            const float *src = &svmDetector[0] + (j * blocks_per_img.height + i) * block_hist_size;
            float *dst = detector_reordered.ptr<float>() + (i * blocks_per_img.width + j) * block_hist_size;
            for (size_t k = 0; k < block_hist_size; ++k)
                dst[k] = src[k];
        }
    size_t descriptor_size = getDescriptorSize();
    free_coef = svmDetector.size() > descriptor_size ? svmDetector[descriptor_size] : 0;
    detector_reordered.copyTo(oclSvmDetector);
}

#define CV_TYPE_NAME_HOG_DESCRIPTOR "opencv-object-detector-hog"


// @img [input] 计算图像img
// @grad [output] 梯度幅度图像`grad`
// @qangle [output] 梯度方向图像`qangle`.
// @paddingTL为需要在原图像img左上角扩增的尺寸，同理paddingBR
// @paddingBR 为需要在img图像右下角扩增的尺寸。
void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
    Size paddingTL, Size paddingBR) const
{
    CV_INSTRUMENT_REGION()

    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
    // padding之后的输出大小
    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
        img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

    Size wholeSize;
    Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* const lut = &_lut(0,0);

    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)BORDER_REFLECT_101;

    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
        wholeSize.width, borderType) - roiofs.x;
    for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* const dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    if (cn == 3)
    {
        int end = gradsize.width + 2;
        xmap -= 1, x = 0;
        for ( ; x < end; ++x)
            xmap[x] *= 3;
        xmap += 1;
    }

    float angleScale = signedGradient ? (float)(nbins/(2.0*CV_PI)) : (float)(nbins/CV_PI);
    for( y = 0; y < gradsize.height; y++ )
    {
        const uchar* imgPtr  = img.ptr(ymap[y]);
        //In case subimage is used ptr() generates an assert for next and prev rows
        //(see http://code.opencv.org/issues/4149)
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = grad.ptr<float>(y);
        uchar* qanglePtr = qangle.ptr(y);

        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
        else
        {
            x = 0;
            for( ; x < width; x++ )
            {
                int x1 = xmap[x];
                float dx0, dy0, dx, dy, mag0, mag;
                const uchar* p2 = imgPtr + xmap[x+1];
                const uchar* p0 = imgPtr + xmap[x-1];

                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

        // computing angles and magnidutes
        cartToPolar( Dx, Dy, Mag, Angle, false );

        // filling the result matrix
        x = 0;

        for( ; x < width; x++ )
        {
            float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
            int hidx = cvFloor(angle);
            angle -= hidx;
            gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;

            if( hidx < 0 )
                hidx += nbins;
            else if( hidx >= nbins )
                hidx -= nbins;

            CV_Assert( (unsigned)hidx < (unsigned)nbins );

            qanglePtr[x*2] = (uchar)hidx;
            hidx++;
            hidx &= hidx < nbins ? -1 : 0;
            qanglePtr[x*2+1] = (uchar)hidx;
        }
    }
}

struct HOGCache
{
	// 1个BlockData结构体是对应的一个block数据。
	// 其中histOfs表示为该block对整个滑动窗口内hog描述算子的贡献那部分向量的起始位置；
	// imgOffset为该block在滑动窗口图片中的坐标(当然是指左上角坐标)
    struct BlockData
    {
        BlockData() :
            histOfs(0), imgOffset()
        { }

        int histOfs;
        Point imgOffset;
    };
    // PixData结构体是对应的block中1个像素点的数据。
    // 其中gradOfs表示该点的梯度幅度在滑动窗口图片梯度幅度图中的位置坐标；
    // qangleOfs表示该点的梯度角度在滑动窗口图片梯度角度图中的位置坐标；
    // histOfs[]表示该像素点对1个或2个或4个cell贡献的hog描述子向量的起始位置坐标（比较抽象，需要看源码才懂）。
    // histWeight[]表示该像素点对1个或2个或4个cell贡献的权重。
    // gradWeight表示该点本身由于处在block中位置的不同因而对梯度直方图贡献也不同，其权值按照二维高斯分布(以block中心为二维高斯的中心)来决定。
    struct PixData
    {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    HOGCache();
    HOGCache(const HOGDescriptor* descriptor,
        const Mat& img, const Size& paddingTL, const Size& paddingBR,
        bool useCache, const Size& cacheStride);
    virtual ~HOGCache() { }
    virtual void init(const HOGDescriptor* descriptor,
        const Mat& img, const Size& paddingTL, const Size& paddingBR,
        bool useCache, const Size& cacheStride);

    Size windowsInImage(const Size& imageSize, const Size& winStride) const;
    Rect getWindow(const Size& imageSize, const Size& winStride, int idx) const;

    const float* getBlock(Point pt, float* buf);
    // 指对block获取到的hog部分描述子进行归一化，其实该归一化有2层，具体看代码。
    virtual void normalizeBlockHistogram(float* histogram) const;

    std::vector<PixData> pixData;
    std::vector<BlockData> blockData;

    bool useCache;
    std::vector<int> ymaxCached;
    Size winSize;
    Size cacheStride;
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;
    Mat_<float> blockCache;
    Mat_<uchar> blockCacheFlags;

    Mat grad, qangle;
    const HOGDescriptor* descriptor;
};

HOGCache::HOGCache() :
    blockHistogramSize(), count1(), count2(), count4()
{
    useCache = false;
    descriptor = 0;
}

HOGCache::HOGCache(const HOGDescriptor* _descriptor,
    const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
    bool _useCache, const Size& _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void HOGCache::init(const HOGDescriptor* _descriptor,
    const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
    bool _useCache, const Size& _cacheStride)
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;
    // 计算输入图像的权值梯度幅度图和角度量化图
    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
    imgoffset = _paddingTL;

    winSize = descriptor->winSize;
    Size blockSize = descriptor->blockSize;
    Size blockStride = descriptor->blockStride;
    Size cellSize = descriptor->cellSize;
    int i, j, nbins = descriptor->nbins;
    // rawBlockSize为block中包含像素点的个数
    int rawBlockSize = blockSize.width*blockSize.height;
    // block的数目
    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
        (winSize.height - blockSize.height)/blockStride.height + 1);
    // cell的数目
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);
    // blockHistogramSize表示一个block中贡献给hog描述子向量的长度
    blockHistogramSize = ncells.width*ncells.height*nbins;

    if( useCache )
    {
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
            (winSize.height/cacheStride.height)+1);

        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        blockCacheFlags.create(cacheSize);

        size_t cacheRows = blockCache.rows;
        ymaxCached.resize(cacheRows);
        for(size_t ii = 0; ii < cacheRows; ii++ )
            ymaxCached[ii] = -1;
    }
    // weights为一个尺寸为blockSize的二维高斯表,下面的代码就是计算二维高斯的系数
    Mat_<float> weights(blockSize);
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

    {
        AutoBuffer<float> di(blockSize.height), dj(blockSize.width);
        float* _di = (float*)di, *_dj = (float*)dj;
        float bh = blockSize.height * 0.5f, bw = blockSize.width * 0.5f;

        for (i = 0; i < blockSize.height; ++i)
        {
            _di[i] = i - bh;
            _di[i] *= _di[i];
        }

        for (j = 0;; j < blockSize.width; ++j)
        {
            _dj[j] = j - bw;
            _dj[j] *= _dj[j];
        }

        for(i = 0; i < blockSize.height; i++)
            for(j = 0; j < blockSize.width; j++)
                weights(i,j) = std::exp(-(_di[i] + _dj[j])*scale);
    }

	// vector<BlockData> blockData;而BlockData为HOGCache的一个结构体成员
	// nblocks.width*nblocks.height表示一个检测窗口中block的个数，
	// 而cacheSize.width*cacheSize.heigh表示一个已经扩充的图片中的block的个数
    blockData.resize(nblocks.width*nblocks.height);
    // vector<PixData> pixData; 同理，Pixdata也为HOGCache中的一个结构体成员
    // rawBlockSize表示每个block中像素点的个数
    // resize表示将其转换成列向量
    // rawBlockSize*3表示的是存储同时对1个cell，2个cell，4个cell的贡献
    pixData.resize(rawBlockSize*3);

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use **pre-computed** blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //

    // count1, count2, count4分别表示block中同时对1个cell，2个cell，4个cell有贡献的像素点的个数。
    count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )
        for( i = 0; i < blockSize.height; i++ )
        {
            PixData* data = 0;
            float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            cellX -= icellX0;
            cellY -= icellY0;

            if( (unsigned)icellX0 < (unsigned)ncells.width &&
               (unsigned)icellX1 < (unsigned)ncells.width )
            {
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                   (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[1] = cellX*(1.f - cellY);
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[2] = (1.f - cellX)*cellY;
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[3] = cellX*cellY;
                }
                else
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            else
            {
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }

                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                   (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = cellX*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
                else
                {
                    data = &pixData[count1++];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = cellX*cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            data->gradOfs = (grad.cols*i + j)*2;
            data->qangleOfs = (qangle.cols*i + j)*2;
            data->gradWeight = weights(i,j);
        }

    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;

    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
}
// 计算一个block中的特征子
const float* HOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

//    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

//    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
//        (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

    if( useCache )
    {
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        Point cacheIdx(pt.x/cacheStride.width,
                       (pt.y/cacheStride.height) % blockCache.rows);
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;
    const float* gradPtr = grad.ptr<float>(pt.y) + pt.x*2;
    const uchar* qanglePtr = qangle.ptr(pt.y) + pt.x*2;

//    CV_Assert( blockHist != 0 );
    memset(blockHist, 0, sizeof(float) * blockHistogramSize);

    const PixData* _pixData = &pixData[0];
    // 统计各个cell中的bin信息
    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        float w = pk.gradWeight*pk.histWeights[0];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* const h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight*pk.histWeights[2];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight*pk.histWeights[3];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }
    // 归一化 block中的Hist
    normalizeBlockHistogram(blockHist);

    return blockHist;
}

void HOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0], sum = 0.0f;
    size_t i = 0, sz = blockHistogramSize;

    for (i = 0 ; i < sz; ++i)
        sum += hist[i]*hist[i];

    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;
    sum = 0.0f;

    for(i = 0; i < sz; ++i)
    {
        hist[i] = std::min(hist[i]*scale, thresh);
        sum += hist[i]*hist[i];
    }

    scale = 1.f/(std::sqrt(sum)+1e-3f), i = 0;

    for ( ; i < sz; ++i)
        hist[i] *= scale;
}

Size HOGCache::windowsInImage(const Size& imageSize, const Size& winStride) const
{
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
        (imageSize.height - winSize.height)/winStride.height + 1);
}

Rect HOGCache::getWindow(const Size& imageSize, const Size& winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX*y;
    return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}

static inline int gcd(int a, int b)
{
    if( a < b )
        std::swap(a, b);
    while( b > 0 )
    {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}


void HOGDescriptor::compute(InputArray _img, std::vector<float>& descriptors,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    CV_INSTRUMENT_REGION()

    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));

    Size imgSize = _img.size();

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(imgSize.width + padding.width*2, imgSize.height + padding.height*2);

    Mat img = _img.getMat();
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);
    // 获取图片中windows的个数
    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();
    descriptors.resize(dsize*nwindows);

    // for each window
    for( size_t i = 0; i < nwindows; i++ )
    {
        float* descriptor = &descriptors[i*dsize];

        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if( src != dst ) memcpy(dst, src, blockHistogramSize * sizeof(float));
        }
    }
}

void HOGDescriptor::detect(const Mat& img,
    std::vector<Point>& hits, std::vector<double>& weights, double hitThreshold,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    CV_INSTRUMENT_REGION()

    hits.clear();
    weights.clear();
    if( svmDetector.empty() )
        return;

    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
        gcd(winStride.height, blockStride.height));

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    std::vector<float> blockHist(blockHistogramSize);


    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                    pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        const float* svmVec = &svmDetector[0];

        int j, k;
        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            const float* vec = cache.getBlock(pt, &blockHist[0]);

            for(k = 0 ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        if( s >= hitThreshold )
        {
            hits.push_back(pt0);
            weights.push_back(s);
        }
    }
}

void HOGDescriptor::detect(const Mat& img, std::vector<Point>& hits, double hitThreshold,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    CV_INSTRUMENT_REGION()

    std::vector<double> weightsV;
    detect(img, hits, weightsV, hitThreshold, winStride, padding, locations);
}

class HOGInvoker : public ParallelLoopBody
{
public:
    HOGInvoker( const HOGDescriptor* _hog, const Mat& _img,
        double _hitThreshold, const Size& _winStride, const Size& _padding,
        const double* _levelScale, std::vector<Rect> * _vec, Mutex* _mtx,
        std::vector<double>* _weights=0, std::vector<double>* _scales=0 )
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
        mtx = _mtx;
    }

    void operator()( const Range& range ) const
    {
        int i, i1 = range.start, i2 = range.end;
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        std::vector<Point> locations;
        std::vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.ptr());
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);
            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));

            mtx->lock();
            for( size_t j = 0; j < locations.size(); j++ )
            {
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                if (scales)
                    scales->push_back(scale);
            }
            mtx->unlock();

            if (weights && (!hitsWeights.empty()))
            {
                mtx->lock();
                for (size_t j = 0; j < locations.size(); j++)
                    weights->push_back(hitsWeights[j]);
                mtx->unlock();
            }
        }
    }

private:
    const HOGDescriptor* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    std::vector<Rect>* vec;
    std::vector<double>* weights;
    std::vector<double>* scales;
    Mutex* mtx;
};


void HOGDescriptor::detectMultiScale(
    InputArray _img, std::vector<Rect>& foundLocations, std::vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const
{
    CV_INSTRUMENT_REGION()

    double scale = 1.;
    int levels = 0;

    Size imgSize = _img.size();
    std::vector<double> levelScale;
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(imgSize.width/scale) < winSize.width ||
            cvRound(imgSize.height/scale) < winSize.height ||
                scale0 <= 1 )
            break;
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);

    if(winStride == Size())
        winStride = blockStride;

    CV_OCL_RUN(_img.dims() <= 2 && _img.type() == CV_8UC1 && scale0 > 1 && winStride.width % blockStride.width == 0 &&
        winStride.height % blockStride.height == 0 && padding == Size(0,0) && _img.isUMat(),
        ocl_detectMultiScale(_img, foundLocations, levelScale, hitThreshold, winStride, finalThreshold, oclSvmDetector,
        blockSize, cellSize, nbins, blockStride, winSize, gammaCorrection, L2HysThreshold, (float)getWinSigma(), free_coef, signedGradient));

    std::vector<Rect> allCandidates;
    std::vector<double> tempScales;
    std::vector<double> tempWeights;
    std::vector<double> foundScales;

    Mutex mtx;
    Mat img = _img.getMat();
    Range range(0, (int)levelScale.size());
    HOGInvoker invoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &mtx, &tempWeights, &tempScales);
    parallel_for_(range, invoker);

    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    foundLocations.clear();
    std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    foundWeights.clear();
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));

    if ( useMeanshiftGrouping )
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
    else
        groupRectangles(foundLocations, foundWeights, (int)finalThreshold, 0.2);
    clipObjects(imgSize, foundLocations, 0, &foundWeights);
}

void HOGDescriptor::detectMultiScale(InputArray img, std::vector<Rect>& foundLocations,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const
{
    CV_INSTRUMENT_REGION()

    std::vector<double> foundWeights;
    detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride,
                padding, scale0, finalThreshold, useMeanshiftGrouping);
}



void HOGDescriptor::detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
    CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
    double hitThreshold, cv::Size winStride, cv::Size padding) const
{
    CV_INSTRUMENT_REGION()

    foundLocations.clear();
    confidences.clear();

    if( svmDetector.empty() || locations.empty())
        return;

    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    // HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);
    HOGCache cache(this, img, padding, padding, true, cacheStride);
    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    std::vector<float> blockHist(blockHistogramSize);


    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        pt0 = locations[i];
        if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
        {
            // out of image
            confidences.push_back(-10.0);
            continue;
        }

        double s = rho;
        const float* svmVec = &svmDetector[0];
        int j, k;

        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            // need to devide this into 4 parts!
            const float* vec = cache.getBlock(pt, &blockHist[0]);

            for(k = 0 ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        confidences.push_back(s);

        if( s >= hitThreshold )
            foundLocations.push_back(pt0);
    }
}

void HOGDescriptor::detectMultiScaleROI(const cv::Mat& img,
    CV_OUT std::vector<cv::Rect>& foundLocations, std::vector<DetectionROI>& locations,
    double hitThreshold, int groupThreshold) const
{
    CV_INSTRUMENT_REGION()

    std::vector<Rect> allCandidates;
    Mutex mtx;

    parallel_for_(Range(0, (int)locations.size()),
                  HOGConfInvoker(this, img, hitThreshold, Size(8, 8),
                                 &locations, &allCandidates, &mtx));

    foundLocations.resize(allCandidates.size());
    std::copy(allCandidates.begin(), allCandidates.end(), foundLocations.begin());
    cv::groupRectangles(foundLocations, groupThreshold, 0.2);
}

}