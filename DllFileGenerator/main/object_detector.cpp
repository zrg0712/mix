#include "object_detector.h"

using namespace std;

namespace PaddleDetection
{
	void detector::load_model()
	{
		paddle_infer::Config config;
		string prog_file = m_model_dir + OS_PATH_SEP + "model.pdmodel";
		string params_file = m_model_dir + OS_PATH_SEP + "model.pdiparams";
		config.SetModel(prog_file, params_file);
		//config.DisableGpu();
		if (m_device == "GPU")
		{
			config.EnableUseGpu(100, 0);
		}
		else
		{
			config.DisableGpu();
			config.EnableMKLDNN();
		}
		config.EnableMemoryOptim();
		config.SetCpuMathLibraryNumThreads(1);
		config.DisableGlogInfo();
		//config.SetMkldnnCacheCapacity(10);
		predictor = paddle_infer::CreatePredictor(config);
	}

	void detector::predict(const vector<cv::Mat> imgs, const double threshold, const int warmup,const string category, vector<vector<int>> offset,
		vector<PaddleDetection::ObjectResult>& result, vector<double>* times)
	{
		if (imgs.size() == 1)
		{
			auto preprocess_start = chrono::steady_clock::now();
			int batch_size = imgs.size();

			// in_data_batch
			vector<float> in_data_all;  // 动态扩展
			vector<float> im_shape_all(batch_size * 2.0f);  // 创建batch_size*2大小的vector，并初始化为0.0
			vector<float> scale_factor_all(batch_size * 2.0f);
			vector<const float*> output_data_list_;
			vector<int> out_bbox_num_data_;
			vector<int> out_mask_data_;

			//in_net img for each batch
			vector<cv::Mat> in_net_img_all(batch_size);  // 存放batch_size大小的图像

			//preprocess image
			for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
			{
				cv::Mat im = imgs.at(bs_idx);
				preprocess_img(im);
				im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];  // image width and height
				im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

				scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];  // Scale factor for image size to origin image size
				scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

				in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
					inputs_.im_data_.end());

				// collect in_net img
				in_net_img_all[bs_idx] = inputs_.in_net_im_;
			}

			auto preprocess_end = std::chrono::steady_clock::now();  // 结束时间
			// prepare input tensor
			auto input_names = predictor->GetInputNames();
			for (const auto& tensor_name : input_names)
			{
				auto in_tensor = predictor->GetInputHandle(tensor_name);
				if (tensor_name == "image" || tensor_name == "x0")
				{
					int rh = inputs_.in_net_shape_[0];
					int rw = inputs_.in_net_shape_[1];
					in_tensor->Reshape({ batch_size,3,rh,rw });
					in_tensor->CopyFromCpu(in_data_all.data());
				}
				else if (tensor_name == "im_shape")
				{
					in_tensor->Reshape({ batch_size,2 });
					in_tensor->CopyFromCpu(im_shape_all.data());
				}
				else if (tensor_name == "scale_factor") {
					in_tensor->Reshape({ batch_size, 2 });
					in_tensor->CopyFromCpu(scale_factor_all.data());
				}
			}
			// Run predictor
			std::vector<std::vector<float>> out_tensor_list;
			std::vector<std::vector<int>> output_shape_list;
			bool is_rbox = false;
			int reg_max = 7;
			int num_class = 1;

			auto inference_start = chrono::steady_clock::now();

			inference_start = std::chrono::steady_clock::now();

			predictor->Run();
			// Get output tensor
			out_tensor_list.clear();
			output_shape_list.clear();
			auto output_names = predictor->GetOutputNames();
			for (int j = 0; j < output_names.size(); j++)
			{
				auto output_tensor = predictor->GetOutputHandle(output_names[j]);
				//cout << "type:" << output_tensor->type() << endl;
				std::vector<int> output_shape = output_tensor->shape();
				int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
					1, std::multiplies<int>());
				output_shape_list.push_back(output_shape);

				std::vector<float> out_data;
				//std::vector<int> out_data;
				out_bbox_num_data_.resize(out_num);
				out_data.resize(out_num);
				output_tensor->CopyToCpu(out_data.data());
				out_tensor_list.push_back(out_data);
			}

			auto inference_end = std::chrono::steady_clock::now();
			auto postprocess_start = std::chrono::steady_clock::now();  // 后处理
			// Postprocessing result
			result.clear();
			is_rbox = output_shape_list[0][output_shape_list[0].size() - 1] % 10 == 0;
			postprocess_img(imgs, result, out_bbox_num_data_, out_tensor_list[0], out_mask_data_, is_rbox);

			auto postprocess_end = std::chrono::steady_clock::now();

			std::chrono::duration<float> preprocess_diff =
				preprocess_end - preprocess_start;
			times->push_back(static_cast<double>(preprocess_diff.count() * 1000));
			std::chrono::duration<float> inference_diff = inference_end - inference_start;
			times->push_back(
				static_cast<double>(inference_diff.count() / 1 * 1000));
			std::chrono::duration<float> postprocess_diff =
				postprocess_end - postprocess_start;
			times->push_back(static_cast<double>(postprocess_diff.count() * 1000));
		}
		else if(category =="circle")
		{
			auto preprocess_start = chrono::steady_clock::now();
			int batch_size = imgs.size();
			std::vector<std::vector<float>> out_tensor_list;
			std::vector<std::vector<int>> output_shape_list;
			bool is_rbox = false;

			std::vector<std::string, std::allocator<std::string>> input_names = predictor->GetInputNames();
			std::unique_ptr<paddle_infer::Tensor, std::default_delete<paddle_infer::Tensor>> in_tensor;
			std::vector<std::string, std::allocator<std::string>> output_names = predictor->GetOutputNames();
			std::unique_ptr<paddle_infer::Tensor, std::default_delete<paddle_infer::Tensor>> output_tensor;
			vector<float> im_shape_all(batch_size * 2.0f);  // 创建batch_size*2大小的vector，并初始化为0.0
			vector<float> scale_factor_all(batch_size * 2.0f);
			for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
			{
				cout << "第 " << bs_idx + 1 << " 张图片" << endl;
				vector<PaddleDetection::ObjectResult> temp_result;
				// in_data_batch
				vector<float> in_data_all;  // 动态扩展
				vector<int> out_bbox_num_data_;


				//in_net img for each batch
				//vector<cv::Mat> in_net_img_all(batch_size);  // 存放batch_size大小的图像

				cv::Mat im = imgs.at(bs_idx);
				preprocess_img(im);
				im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];  // image width and height
				im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

				scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];  // Scale factor for image size to origin image size
				scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

				in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());

				for (const auto& tensor_name : input_names)
				{
					in_tensor = predictor->GetInputHandle(tensor_name);
					if (tensor_name == "image" || tensor_name == "x0")
					{
						int rh = inputs_.in_net_shape_[0];
						int rw = inputs_.in_net_shape_[1];
						in_tensor->Reshape({ 1,3,rh,rw });
						in_tensor->CopyFromCpu(in_data_all.data());
					}
					else if (tensor_name == "im_shape")
					{
						in_tensor->Reshape({ 1,2 });
						in_tensor->CopyFromCpu(im_shape_all.data());
					}
					else if (tensor_name == "scale_factor")
					{
						in_tensor->Reshape({ 1, 2 });
						in_tensor->CopyFromCpu(scale_factor_all.data());
					}
				}

				bool is_rbox = false;
				int reg_max = 7;
				int num_class = 1;
				//auto inference_start = chrono::steady_clock::now();

				// 预测阶段
				//inference_start = std::chrono::steady_clock::now();

				predictor->Run();
				// Get output tensor
				//out_tensor_list.clear();
				//output_shape_list.clear();

				output_tensor = predictor->GetOutputHandle(output_names[0]);
				//cout << "type:" << output_tensor->type() << endl;
				std::vector<int> output_shape = output_tensor->shape();
				int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
					1, std::multiplies<int>());
				output_shape_list.push_back(output_shape);

				std::vector<float> out_data;
				//std::vector<int> out_data;
				out_bbox_num_data_.resize(out_num);
				out_data.resize(out_num);
				output_tensor->CopyToCpu(out_data.data());
				// TODO:若小图推测有进行resize,要还原成原来的尺寸，可以用FLAGS控制
				// 
				int x_offset = offset[bs_idx][0];
				int y_offset = offset[bs_idx][1];

				int boxes_num = out_bbox_num_data_.size() / 6;
				for (int i = 0; i < boxes_num; i++)
				{
					if (out_data[4 + i * 6] > 0.4)
					{
						int class_id = static_cast<int>(round(out_data[5 + i * 6]));
						// Confidence score
						float score = out_data[4 + i * 6];
						float cx = out_data[0 + i * 6] + x_offset;
						float cy = out_data[1 + i * 6] + y_offset;
						float width = out_data[2 + i * 6];
						float height = out_data[3 + i * 6];

						int xmin = std::max(int(cx - (width / 2)), 0);
						int ymin = std::max(int(cy - (height / 2)), 0);
						int xmax = std::min(int(cx + (width / 2)), 3072);
						int ymax = std::min(int(cy + (height / 2)), 3072);

						PaddleDetection::ObjectResult result_item;
						result_item.rect = { xmin, ymin, xmax, ymax };
						result_item.class_id = class_id;
						result_item.confidence = score;

						result.push_back(result_item);
					}
				}
			}
			if (result.size() < 5000)
			{
				PaddleDetection::nms(result, 0.25);
			}
			else
			{
				vector<PaddleDetection::ObjectResult> temp_result;
				temp_result = PaddleDetection::nms_plus(result, 0.25);
				nms(temp_result, 0.25);
				result.clear();
				result.insert(result.end(), temp_result.begin(), temp_result.end());
			}

			auto preprocess_end = std::chrono::steady_clock::now();
			std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
			times->push_back(static_cast<double>(preprocess_diff.count() * 1000));
		}
		else
		{
			auto preprocess_start = chrono::steady_clock::now();
			int batch_size = imgs.size();
			std::vector<std::vector<float>> out_tensor_list;
			std::vector<std::vector<int>> output_shape_list;
			bool is_rbox = false;

			std::vector<std::string, std::allocator<std::string>> input_names = predictor->GetInputNames();
			std::unique_ptr<paddle_infer::Tensor, std::default_delete<paddle_infer::Tensor>> in_tensor;
			std::vector<std::string, std::allocator<std::string>> output_names = predictor->GetOutputNames();
			std::unique_ptr<paddle_infer::Tensor, std::default_delete<paddle_infer::Tensor>> output_tensor;
			vector<float> im_shape_all(batch_size * 2.0f);  // 创建batch_size*2大小的vector，并初始化为0.0
			vector<float> scale_factor_all(batch_size * 2.0f);
			for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
			{
				cout << "第 " << bs_idx + 1 << " 张图片" << endl;
				vector<PaddleDetection::ObjectResult> temp_result;
				// in_data_batch
				vector<float> in_data_all;  // 动态扩展
				vector<int> out_bbox_num_data_;

				cv::Mat im = imgs.at(bs_idx);
				preprocess_img(im);
				im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];  // image width and height
				im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

				scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];  // Scale factor for image size to origin image size
				scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

				in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());

				for (const auto& tensor_name : input_names)
				{
					in_tensor = predictor->GetInputHandle(tensor_name);
					if (tensor_name == "image" || tensor_name == "x0")
					{
						int rh = inputs_.in_net_shape_[0];
						int rw = inputs_.in_net_shape_[1];
						in_tensor->Reshape({ 1,3,rh,rw });
						in_tensor->CopyFromCpu(in_data_all.data());
					}
					else if (tensor_name == "im_shape")
					{
						in_tensor->Reshape({ 1,2 });
						in_tensor->CopyFromCpu(im_shape_all.data());
					}
					else if (tensor_name == "scale_factor")
					{
						in_tensor->Reshape({ 1, 2 });
						in_tensor->CopyFromCpu(scale_factor_all.data());
					}
				}

				//bool is_rbox = false;
				int reg_max = 7;
				int num_class = 1;
				//auto inference_start = chrono::steady_clock::now();

				// 预测阶段
				//inference_start = std::chrono::steady_clock::now();

				predictor->Run();
				// Get output tensor
				//out_tensor_list.clear();
				//output_shape_list.clear();

				output_tensor = predictor->GetOutputHandle(output_names[0]);
				//cout << "type:" << output_tensor->type() << endl;
				std::vector<int> output_shape = output_tensor->shape();
				int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
					1, std::multiplies<int>());
				output_shape_list.push_back(output_shape);

				std::vector<float> out_data;
				//std::vector<int> out_data;
				out_bbox_num_data_.resize(out_num);
				out_data.resize(out_num);
				output_tensor->CopyToCpu(out_data.data());
				// TODO:若小图推测有进行resize,要还原成原来的尺寸，可以用FLAGS控制
				// 
				int x_offset = offset[bs_idx][0];
				int y_offset = offset[bs_idx][1];

				int boxes_num = out_bbox_num_data_.size() / 10;
				for (int i = 0; i < boxes_num; i++)
				{
					if (out_data[1 + i * 10] > 0.2)
					{
						int x1 = out_data[2 + i * 10] + x_offset;
						int y1 = out_data[3 + i * 10] + y_offset;
						int x2 = out_data[4 + i * 10] + x_offset;
						int y2 = out_data[5 + i * 10] + y_offset;
						int x3 = out_data[6 + i * 10] + x_offset;
						int y3 = out_data[7 + i * 10] + y_offset;
						int x4 = out_data[8 + i * 10] + x_offset;
						int y4 = out_data[9 + i * 10] + y_offset;

						// Class id
						int class_id = static_cast<int>(round(out_data[0 + i * 10]));
						// Confidence score
						float score = out_data[1 + i * 10];

						PaddleDetection::ObjectResult result_item;
						result_item.rect = { x1, y1, x2, y2,x3, y3, x4, y4 };
						result_item.class_id = class_id;
						result_item.confidence = score;

						result.push_back(result_item);
					}
				}
			}
			if (result.size() < 5000)
			{
				PaddleDetection::strip_nms(result, 0.25);
			}
			else
			{
				vector<PaddleDetection::ObjectResult> temp_result;
				temp_result = PaddleDetection::strip_nms_plus(result, 0.25);
				strip_nms(temp_result, 0.25);
				result.clear();
				result.insert(result.end(), temp_result.begin(), temp_result.end());
			}
			auto preprocess_end = std::chrono::steady_clock::now();
			std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
			times->push_back(static_cast<double>(preprocess_diff.count() * 1000));
		}
	}

	void detector::postprocess_img(const std::vector<cv::Mat> mats,
		std::vector<PaddleDetection::ObjectResult>& result,
		std::vector<int> bbox_num, std::vector<float> output_data_,
		std::vector<int> output_mask_data_, bool is_rbox)
	{
		result.clear();
		int start_idx = 0;
		//int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
		int total_num = bbox_num.size() / 6;
		int out_mask_dim = -1;
		if (config_.mask_) {
			out_mask_dim = output_mask_data_.size() / total_num;
		}

		for (int im_id = 0; im_id < mats.size(); im_id++)
		{
			for (int j = start_idx; j < total_num; j++)
			{
				// Class id
				int class_id = static_cast<int>(round(output_data_[5 + j * 6]));
				// Confidence score
				float score = output_data_[4 + j * 6];
				float cx = output_data_[0 + j * 6];
				float cy = output_data_[1 + j * 6];
				float width = output_data_[2 + j * 6];
				float height = output_data_[3 + j * 6];

				int xmin = cx - (width / 2);
				int ymin = cy - (height / 2);
				int xmax = cx + (width / 2);
				int ymax = cy + (height / 2);


				PaddleDetection::ObjectResult result_item;
				result_item.rect = { xmin, ymin, xmax, ymax };
				result_item.class_id = class_id;
				result_item.confidence = score;

				if (result_item.confidence > 0.5)
				{
					result.push_back(result_item);
				}

			}
		}
		PaddleDetection::nms(result, 0.4);
	}

	void detector::strip_postprocess_img(const std::vector<cv::Mat> mats,
		std::vector<PaddleDetection::ObjectResult>& result,
		std::vector<std::vector<float>> output_data_, bool is_rbox)
	{
		result.clear();
		int start_idx = 0;
		for (int start_idx = 0; start_idx < mats.size(); start_idx++)
		{
			int total_num = output_data_[start_idx].size() / 10;

			for (int j = 0; j < total_num; j++)
			{
				// Class id
				int class_id = static_cast<int>(round(output_data_[start_idx][0 + j * 10]));
				// Confidence score
				float score = output_data_[start_idx][1 + j * 10];

				int x1 = output_data_[start_idx][2 + j * 10];
				int y1 = output_data_[start_idx][3 + j * 10];
				int x2 = output_data_[start_idx][4 + j * 10];
				int y2 = output_data_[start_idx][5 + j * 10];
				int x3 = output_data_[start_idx][6 + j * 10];
				int y3 = output_data_[start_idx][7 + j * 10];
				int x4 = output_data_[start_idx][8 + j * 10];
				int y4 = output_data_[start_idx][9 + j * 10];

				PaddleDetection::ObjectResult result_item;
				result_item.rect = { x1, y1, x2, y2,x3, y3, x4, y4 };
				result_item.class_id = class_id;
				result_item.confidence = score;

				if (result_item.confidence > 0.2)
				{
					result.push_back(result_item);
				}
			}
		}
		// 添加strip_nms
 		PaddleDetection::strip_nms(result, 0.1);
	}

	//void detector::cirlce_postprocess_img(const std::vector<cv::Mat> mats,
	//	std::vector<PaddleDetection::ObjectResult>& result,
	//	std::vector<std::vector<float>> output_data_, bool is_rbox)
	//{
	//	result.clear();
	//	int start_idx = 0;
	//	for (int start_idx = 0; start_idx < mats.size(); start_idx++)
	//	{
	//		int total_num = output_data_[start_idx].size() / 6;

	//		for (int j = 0; j < total_num; j++)
	//		{
	//			// Class id
	//			int class_id = static_cast<int>(round(output_data_[start_idx][5 + j * 6]));
	//			// Confidence score
	//			float score = output_data_[start_idx][4 + j * 6];
	//			float cx = output_data_[start_idx][0 + j * 6];
	//			float cy = output_data_[start_idx][1 + j * 6];
	//			float width = output_data_[start_idx][2 + j * 6];
	//			float height = output_data_[start_idx][3 + j * 6];

	//			int xmin = std::max(int(cx - (width / 2)), 0);
	//			int ymin = std::max(int(cy - (height / 2)), 0);
	//			int xmax = std::min(int(cx + (width / 2)), 3072);
	//			int ymax = std::min(int(cy + (height / 2)), 3072);

	//			PaddleDetection::ObjectResult result_item;
	//			result_item.rect = { xmin, ymin, xmax, ymax };
	//			result_item.class_id = class_id;
	//			result_item.confidence = score;

	//			if (result_item.confidence > 0.4)
	//			{
	//				result.push_back(result_item);
	//			}
	//		}
	//	}
	//	// 添加small_nms
	//	PaddleDetection::nms(result, 0.25);
	//}

	void detector::preprocess_img(const cv::Mat& ori_im)
	{
		// Clone the image : keep the original mat for postprocess
		cv::Mat im = ori_im.clone();
		cv::cvtColor(im, im, cv::COLOR_BGR2RGB);  // 改变图像通道
		preprocessor_.Run(&im, &inputs_);
	}

	cv::Mat VisualizeResult(const cv::Mat& img,
		const std::vector<PaddleDetection::ObjectResult>& results,
		const std::vector<std::string>& lables, const bool is_rbox = false)
	{
		cv::Mat vis_img = img.clone();
		int img_h = vis_img.rows;
		int img_w = vis_img.cols;

		int c1 = 0;
		int c2 = 255;
		int c3 = 0;
		cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
		int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
		double font_scale = 2.0f;
		float thickness = 1.5;
		for (int i = 0; i < results.size(); ++i)
		{
			cv::Point origin;

			if (is_rbox)
			{
				// Draw object, text, and background
				for (int k = 0; k < 4; k++) {
					cv::Point pt1 = cv::Point(results[i].rect[(k * 2) % 8],
						results[i].rect[(k * 2 + 1) % 8]);
					cv::Point pt2 = cv::Point(results[i].rect[(k * 2 + 2) % 8],
						results[i].rect[(k * 2 + 3) % 8]);
					cv::line(vis_img, pt1, pt2, roi_color, 1.5);
				}
			}
			else
			{
				int w = results[i].rect[2] - results[i].rect[0];
				int h = results[i].rect[3] - results[i].rect[1];
				cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
				// Draw roi object, text, and background
				cv::rectangle(vis_img, roi, roi_color, 2);
			}

			//origin.x = results[i].rect[0];
			//origin.y = results[i].rect[1];

			// Configure text background
			//cv::Rect text_back =
			//	cv::Rect(results[i].rect[0], results[i].rect[1] - text_size.height,
			//		text_size.width, text_size.height);
			//// Draw text, and background
			//cv::rectangle(vis_img, text_back, roi_color, -1);
			//cv::putText(vis_img, text, origin, font_face, font_scale,
			//	cv::Scalar(255, 255, 255), thickness);
		}
		std::string cout = "Number =  " + to_string(results.size());
		cv::Point origin(img_h / 2, 200);
		cv::putText(vis_img, cout, origin, font_face, font_scale,
		cv::Scalar(255, 0, 0), thickness);
		return vis_img;
	}

}  // namespace PaddleDetection
