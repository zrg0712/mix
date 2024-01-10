#include "detect_utils.h"

inline void printInferLog(std::vector<double> det_time, int img_num) {
	cout << "----------------------- Infer Info -----------------------" << endl;
	cout << "Total number of predicted data: " << img_num << endl;
	cout << " and total time spent(ms): " << std::accumulate(det_time.begin(), det_time.end(), 0) << endl;
	cout << "preproce_time(ms): " << det_time[0] / img_num
		<< ", inference_time(ms): " << det_time[1] / img_num
		<< ", postprocess_time(ms): " << det_time[2] / img_num << endl;
}

vector<PaddleDetection::ObjectResult> classicication_disc(const vector<string> img_path, PaddleDetection::detector& det, string image_file, string category, string output_dir)
{
	vector<cv::Mat> batch_imgs;
	vector<double> det_t = { 0,0,0 };
	string image_file_path = img_path.at(0);
	cv::Mat im = cv::imread(image_file_path, 1);
	batch_imgs.insert(batch_imgs.end(), im);

	vector<PaddleDetection::ObjectResult> result;
	vector<double> det_times;
	bool is_rbox = false;

	std::vector<int> off_set = { 0 };
	std::vector<vector<int>> offsets;
	offsets.push_back(off_set);

	// threshold 0.5
	det.predict(batch_imgs, 0.5, 0, category, offsets, result, &det_times);
	auto labels = det.getLabelList();

	im = batch_imgs[0];
	vector<PaddleDetection::ObjectResult> im_result;
	int detect_num = 0;

	for (int i = 0; i < result.size(); i++)
	{
		PaddleDetection::ObjectResult item = result[i];
		if (item.class_id == -1)
		{
			continue;
		}
		item.rect[0] = int(float(item.rect[0]) * 3072.0f / 640.0f);
		item.rect[1] = int(float(item.rect[1]) * 3072.0f / 640.0f);
		item.rect[2] = int(float(item.rect[2]) * 3072.0f / 640.0f);
		item.rect[3] = int(float(item.rect[3]) * 3072.0f / 640.0f);
		detect_num += 1;
		im_result.push_back(item);

	}
	cout << img_path.at(0) << " The number of detected box:	" << detect_num << endl;
	cv::Mat vis_img = PaddleDetection::VisualizeResult(im, im_result, labels, is_rbox);

	string output_path(output_dir);

	string filename = PaddleDetection::getFileName(image_file);

	output_path = output_path + "\\" + filename;
	cv::imwrite(output_path, vis_img);
	cout << "Visualized output saved as " << output_path.c_str() << endl;

	det_t[0] += det_times[0];
	det_t[1] += det_times[1];
	det_t[2] += det_times[2];
	det_times.clear();

	printInferLog(det_t, img_path.size());

	return im_result;
}

vector<PaddleDetection::ObjectResult> multiple_strip_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path)
{
	PaddleDetection::detector myDetector(model_path, device);
	std::vector<std::string> all_img_paths;
	std::vector<cv::String> cv_all_img_paths;
	cv::glob(temp_path, cv_all_img_paths);
	for (const auto& img_path : cv_all_img_paths)
	{
		all_img_paths.push_back(img_path);
	}

	labels = myDetector.getLabelList();

	std::vector<double> det_t = { 0,0,0 };
	std::vector<cv::Mat> batch_imgs;
	std::vector<int> off_set;
	std::vector<vector<int>> offsets;
	// 增加文件名偏移量
	for (int bs = 0; bs < all_img_paths.size(); bs++) {
		std::string image_file_path = all_img_paths.at(bs);
		cv::Mat im = cv::imread(image_file_path, 1);
		batch_imgs.insert(batch_imgs.end(), im);
		off_set = PaddleDetection::get_offset(image_file_path);
		offsets.push_back(off_set);
		off_set.clear();
	}

	// Store all detected result
	std::vector<PaddleDetection::ObjectResult> result;
	std::vector<double> det_times;
	bool is_rbox = false;

	// threshold 0.5
	myDetector.predict(batch_imgs, 0.4, 0, category, offsets, result, &det_times);

	det_t[0] += det_times[0];
	det_t[1] += 0;
	det_t[2] += 0;
	det_times.clear();

	printInferLog(det_t, batch_imgs.size());

	return result;
}

vector<PaddleDetection::ObjectResult> multiple_circle_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path)
{
	PaddleDetection::detector myDetector(model_path, device);
	std::vector<std::string> all_img_paths;
	std::vector<cv::String> cv_all_img_paths;
	cv::glob(temp_path, cv_all_img_paths);
	for (const auto& img_path : cv_all_img_paths)
	{
		all_img_paths.push_back(img_path);
	}

	labels = myDetector.getLabelList();

	std::vector<double> det_t = { 0,0,0 };
	std::vector<cv::Mat> batch_imgs;
	std::vector<int> off_set;
	std::vector<vector<int>> offsets;
	// 增加文件名偏移量
	for (int bs = 0; bs < all_img_paths.size(); bs++) {
		std::string image_file_path = all_img_paths.at(bs);
		cv::Mat im = cv::imread(image_file_path, 1);
		batch_imgs.insert(batch_imgs.end(), im);
		off_set = PaddleDetection::get_offset(image_file_path);
		offsets.push_back(off_set);
		off_set.clear();
	}

	// Store all detected result
	std::vector<PaddleDetection::ObjectResult> result;
	std::vector<double> det_times;
	bool is_rbox = false;

	// threshold 0.5
	myDetector.predict(batch_imgs, 0.4, 0, category, offsets, result, &det_times);

	det_t[0] += det_times[0];
	det_t[1] += 0;
	det_t[2] += 0;
	det_times.clear();

	printInferLog(det_t, batch_imgs.size());

	return result;
}

vector<PaddleDetection::ObjectResult> single_circle_detect(string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path)
{
	PaddleDetection::detector myDetector(model_path, device);
	std::vector<std::string> all_img_paths;
	std::vector<cv::String> cv_all_img_paths;
	cv::glob(temp_path, cv_all_img_paths);
	for (const auto& img_path : cv_all_img_paths)
	{
		all_img_paths.push_back(img_path);
	}

	labels = myDetector.getLabelList();

	std::vector<double> det_t = { 0,0,0 };
	std::vector<cv::Mat> batch_imgs;
	std::vector<int> off_set;
	std::vector<vector<int>> offsets;
	// 增加文件名偏移量
	for (int bs = 0; bs < all_img_paths.size(); bs++) {
		std::string image_file_path = all_img_paths.at(bs);
		cv::Mat im = cv::imread(image_file_path, 1);
		batch_imgs.insert(batch_imgs.end(), im);
		off_set = PaddleDetection::get_offset(image_file_path);
		offsets.push_back(off_set);
		off_set.clear();
	}

	// Store all detected result
	std::vector<PaddleDetection::ObjectResult> result;
	std::vector<double> det_times;
	bool is_rbox = false;

	// threshold 0.5
	myDetector.predict(batch_imgs, 0.4, 0, category, offsets, result, &det_times);

	det_t[0] += det_times[0];
	det_t[1] += 0;
	det_t[2] += 0;
	det_times.clear();

	printInferLog(det_t, batch_imgs.size());

	return result;
}

vector<PaddleDetection::ObjectResult> single_strip_detect(const string temp_path, std::vector<std::string>& labels, string device, string category, const string model_path)
{
	PaddleDetection::detector myDetector(model_path, device);
	std::vector<std::string> all_img_paths;
	std::vector<cv::String> cv_all_img_paths;
	cv::glob(temp_path, cv_all_img_paths);
	for (const auto& img_path : cv_all_img_paths)
	{
		all_img_paths.push_back(img_path);
	}

	labels = myDetector.getLabelList();

	std::vector<double> det_t = { 0,0,0 };
	std::vector<cv::Mat> batch_imgs;
	std::vector<int> off_set;
	std::vector<vector<int>> offsets;
	// 增加文件名偏移量
	for (int bs = 0; bs < all_img_paths.size(); bs++) {
		std::string image_file_path = all_img_paths.at(bs);
		cv::Mat im = cv::imread(image_file_path, 1);
		batch_imgs.insert(batch_imgs.end(), im);
		off_set = PaddleDetection::get_offset(image_file_path);
		offsets.push_back(off_set);
		off_set.clear();
	}

	// Store all detected result
	std::vector<PaddleDetection::ObjectResult> result;
	std::vector<double> det_times;
	bool is_rbox = false;

	// threshold 0.5
	myDetector.predict(batch_imgs, 0.4, 0, category, offsets, result, &det_times);

	det_t[0] += det_times[0];
	det_t[1] += 0;
	det_t[2] += 0;
	det_times.clear();

	printInferLog(det_t, batch_imgs.size());

	return result;

}
