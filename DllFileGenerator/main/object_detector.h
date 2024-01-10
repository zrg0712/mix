#pragma once

#include <vector>
#include <ctime>
#include <string>
#include <math.h>
#include <paddle_inference_api.h>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "preprocess.h"
#include "config_parser.h"

#define OS_PATH_SEP "\\"

using namespace std;

namespace PaddleDetection
{
	cv::Mat VisualizeResult(const cv::Mat& img,
		const std::vector<PaddleDetection::ObjectResult>& results,
		const std::vector<std::string>& lables, const bool is_rbox);
	class detector
	{
	public:
		explicit detector(const string& model_dir, const string device)
		{
			this->m_device = device;
			this->m_model_dir = model_dir;
			load_model();
			config_.load_config(model_dir);
			preprocessor_.Init(config_.preprocess_info_);
		}

		void load_model();

		void predict(const vector<cv::Mat> imgs, const double threshold, const int warmup, const string category, vector<vector<int>> offset,
			vector<PaddleDetection::ObjectResult>& resutl, vector<double>* times = nullptr);

		void preprocess_img(const cv::Mat& ori_im);

		void postprocess_img(const std::vector<cv::Mat> mats,
			std::vector<PaddleDetection::ObjectResult>& result,
			std::vector<int> bbox_num, std::vector<float> output_data_,
			std::vector<int> output_mask_data_, bool is_rbox);

		//void cirlce_postprocess_img(const std::vector<cv::Mat> mats,
		//	std::vector<PaddleDetection::ObjectResult>& result,
		//	std::vector<std::vector<float>> output_data_, bool is_rbox);

		void strip_postprocess_img(const std::vector<cv::Mat> mats,
			std::vector<PaddleDetection::ObjectResult>& result,
			std::vector<std::vector<float>> output_data_, bool is_rbox);

		shared_ptr<paddle_infer::Predictor> predictor;

		// Get Model Label list
		const std::vector<std::string>& getLabelList() const {
			return config_.label_list_;
		}

	private:
		string m_device;
		string m_model_dir;
		ImageBlob inputs_;
		ConfigParser config_;
		Preprocessor preprocessor_;
	};

}  // namespace PaddleDetection
