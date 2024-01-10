#pragma once
#include <iostream>
#include <string>
#include <gflags/gflags.h>
#include <Windows.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <numeric>
#include "object_detector.h"

#include <paddle_inference_api.h>

void printInferLog(std::vector<double> det_time, int img_num);

vector<PaddleDetection::ObjectResult> classicication_disc(const vector<string> img_path, PaddleDetection::detector& det, string image_file, string category, string output_dir = "output");

vector<PaddleDetection::ObjectResult> multiple_strip_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path = "strip_detect");

vector<PaddleDetection::ObjectResult> multiple_circle_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path = "small_detect");

vector<PaddleDetection::ObjectResult> single_circle_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path = "circle_detect");

vector<PaddleDetection::ObjectResult> single_strip_detect(const string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path = "strip_detect");