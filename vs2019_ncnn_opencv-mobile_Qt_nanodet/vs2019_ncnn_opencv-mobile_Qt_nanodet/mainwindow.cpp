#include "mainwindow.h"

#include<QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowTitle("NanoDetNcnn");

    // 创建摄像头和摄像头捕获对象，设置成捕获到缓冲区，不然会拼命存图到硬盘
    QCamera* cam = new QCamera;
    cam->setCaptureMode(QCamera::CaptureStillImage);
    QCameraImageCapture* cap = new QCameraImageCapture(cam);
    cap->setCaptureDestination(QCameraImageCapture::CaptureToBuffer);

    // 加载模型
    nanoDetNcnn = new NanoDetNcnn();
    nanoDetNcnn->loadModel(0, 0);

    QObject::connect(cap, &QCameraImageCapture::imageCaptured, [=](int id, QImage img) {
        cv::Mat mat = QImageToMat(img); // QImage转成Mat
        mat = nanoDetNcnn->detectDraw(mat); // ncnn推理
        QImage image = MatToQImage(mat); // Mat转回去QImage
        ui.label->setPixmap(QPixmap::fromImage(image)); // 写入QLabel显示
        });

    QObject::connect(cap, &QCameraImageCapture::readyForCaptureChanged, [=](bool state) {
        if (state == true) {
            cam->searchAndLock();
            cap->capture();
            cam->unlock();
        }
        });

    cam->start();

    // 设置分辨率
    QCameraViewfinderSettings set;
    set.setResolution(640, 480);
    cam->setViewfinderSettings(set);

}

void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    nanoDetNcnn->loadModel(index, 0);
}

inline cv::Mat MainWindow::QImageToMat(const QImage& image)
{
    QImage   swapped = image;
    if (image.format() == QImage::Format_RGB32) {
        swapped = swapped.convertToFormat(QImage::Format_RGB888);
    }
    swapped = swapped.rgbSwapped();

    return cv::Mat(swapped.height(), swapped.width(),
        CV_8UC3,
        const_cast<uchar*>(swapped.bits()),
        static_cast<size_t>(swapped.bytesPerLine())
    ).clone();
}

inline QImage MainWindow::MatToQImage(const cv::Mat& mat)
{
    QImage image(mat.data,
        mat.cols, mat.rows,
        static_cast<int>(mat.step),
        QImage::Format_RGB888);

    return image.rgbSwapped();
}