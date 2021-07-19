#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mainwindow.h"

#include<QImage>
#include<QCamera>
#include<QCameraImageCapture>

#include<opencv2/opencv.hpp>

#include"nanodetncnn.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = Q_NULLPTR);

private:
    inline cv::Mat QImageToMat(const QImage& image);
    inline QImage MatToQImage(const cv::Mat& mat);

private slots:
    void on_comboBox_currentIndexChanged(int index);

private:
    Ui::MainWindowClass ui;
    NanoDetNcnn* nanoDetNcnn;
};
