#include "nanodetncnn.h"

NanoDetNcnn::NanoDetNcnn()
{
    g_nanodet = 0;
}

int NanoDetNcnn::draw_fps(int w, int h, cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "%dx%d FPS=%.2f", w, h, avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

cv::Mat NanoDetNcnn::detectDraw(const cv::Mat& img)
{
    cv::Mat rgb = img.clone();
    int w = rgb.cols;
    int h = rgb.rows;

    // 将RGB图喂入ncnn进行推理，绘制bbox和fps
    {
        ncnn::MutexLockGuard g(lock);
        std::vector<Object> objects;
        g_nanodet->detect(rgb, objects);
        g_nanodet->draw(rgb, objects);
    }
    draw_fps(w, h, rgb);

    return rgb;
}

bool NanoDetNcnn::loadModel(int modelid, int cpugpu)
{
    const char* modeltypes[] =
    {
        "m",
        "m-416",
        "g",
        "ELite0_320",
        "ELite1_416",
        "ELite2_512",
        "RepVGG-A0_416"
    };

    const int target_sizes[] =
    {
        320,
        416,
        416,
        320,
        416,
        512,
        416
    };

    const float mean_vals[][3] =
    {
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {127.f, 127.f, 127.f},
        {127.f, 127.f, 127.f},
        {127.f, 127.f, 127.f},
        {103.53f, 116.28f, 123.675f}
    };

    const float norm_vals[][3] =
    {
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
        {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},
        {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},
        {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},
        {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}
    };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);
        if (!g_nanodet)
            g_nanodet = new NanoDet;
        g_nanodet->load(modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
    }
    return true;
}

