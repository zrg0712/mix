#pragma once

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include <direct.h>
#include <sstream>
#include <string>
#include <utility>

#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std;

namespace PaddleDetection {
	struct ObjectResult
	{
		// Rectangle coordinates of detected object:left, right, top, down
		std::vector<int> rect;
		int class_id;
		float confidence;
		std::vector<int> mask;
	};
	void nms(std::vector<ObjectResult>& input_boxes, float nms_threshold);
	vector<ObjectResult> nms_plus(std::vector<ObjectResult>& input_boxes, float nms_threshold);
	void nms_with_thread(vector<ObjectResult> input_boxes, vector<ObjectResult>& result);
	void strip_nms(vector<ObjectResult>& input_boxes, float nms_threshold);
	void strip_nms_with_thread(vector<ObjectResult> input_boxes, vector<ObjectResult>& result);
	vector<ObjectResult> strip_nms_plus(vector<ObjectResult>& input_boxes, float nms_threshold);
	// 图像划分
	vector<cv::Mat> split_image(vector<vector<int>> rect, string img_path, int offset, string save_dir, vector<string>& save_patch_path);
	// 判断文件是否存在
	bool path_exists(const std::string& path);
	// 创建文件
	void mkDir(const std::string& path);
	// 子图切割
	void image_slice(cv::Mat img, string temp_save_path, int size, float overlap_ratio);
	// 获得一系列boxes
	vector<vector<int>> get_slice_boxes(int height, int width, int img_size, float overlap_ratio);
	// 获取文件名中的目标偏移量
	vector<int> get_offset(string img_file_path);
	// 获取文件名123
	string getFileName(const string& path);

	void del_file(string dir);
}  // namespace PaddleDetection