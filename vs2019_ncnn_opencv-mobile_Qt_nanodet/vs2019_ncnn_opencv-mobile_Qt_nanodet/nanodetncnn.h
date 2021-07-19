#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class NanoDetNcnn
{
public:
    NanoDetNcnn();

	cv::Mat detectDraw(const cv::Mat& img);
	bool loadModel(int modelid, int cpugpu);

    int draw_fps(int w, int h, cv::Mat& rgb);

    NanoDet* g_nanodet;
    ncnn::Mutex lock;
};




