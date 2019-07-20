#include <iostream>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "dataStructures.h"

using namespace std;

void detectObjects2()
{
    // load image from file
    cv::Mat img = cv::imread("../images/s_thrun.jpg");

    // load class names from file
    string yoloBasePath = "../dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";   //the classes name yolo can classify
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";    //the yolo network definition
    string yoloModelWeights = yoloBasePath + "yolov3.weights";  //the yolo pre-trained weight parameters

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line); //read all classes name
    
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);

    //backendId 表示后台计算id，
    //DNN_BACKEND_INFERENCE_ENGINE表示使用intel的预测推断库Intel's Inference Engine computational backend，加速性能明显，能提升一个数量级；需要单独安装相关库OpenVINO
    //DNN_BACKEND_OPENCV 一般情况都是使用opencv dnn作为后台计算
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); //set backend to opencv

    //DNN_TARGET_CPU表示在CPU设备上使用
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);   //set target to cpu

    // generate 4D blob from input image, the image will be resize.
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    //scalar with mean values which are subtracted from channels.
    cv::Scalar mean = cv::Scalar(0,0,0);
    //flag which indicates that swap first and last channels in 3-channel image is necessary. 
    bool swapRB = false;
    //flag which indicates whether image will be cropped after resize or not 
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        names[i] = layersNames[outLayers[i] - 1];
        std::cout<< "names[" <<i<<"]: "<< names[i] <<endl;
    }

    // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names); //forward to get the result

    // Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.30;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;

        //iterate every output: cv::MAT::netOutput[i];
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            //get scores for each class.
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;
            
            // Get the value and location of the maximum score
            // minMaxLoc, Finds global minimum and maximum matrix elements and returns their values with locations. 
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);


            //filter the box with small confidence
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top
                
                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // perform non-maxima suppression
    float nmsThreshold = 0.5;  // Non-maximum suppression threshold
    vector<int> indices;

    //cv::dnn::NMSBoxes, Performs non maximum suppression given boxes and corresponding scores. 
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        
        bBoxes.push_back(bBox);
    }
    
    
    // show results
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    detectObjects2();
}