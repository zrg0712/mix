#include "detect_implement.h"
#include "detect_utils.h"

using namespace std;

int detectMain(string image_file, string device, string category="circle")
{
	string output_dir = "output";
	if (!PaddleDetection::path_exists(output_dir)) {
		PaddleDetection::mkDir(output_dir);
	}

	cout << "use " << device << " to infer" << endl;
	string model_dir = "disc_classification";
	PaddleDetection::detector myDetector(model_dir,device);

	vector<string> img_path;
	img_path.push_back(image_file);

	vector<PaddleDetection::ObjectResult> result;
	result = classicication_disc(img_path, myDetector, image_file, category, output_dir);
	// 创建临时文件
	string temp_save_path = "temp_patches";
	// 根据分类结果检测
	if (result.size() == 1 && category == "circle")
	{
		std::cout << "Single Circle Detect" << std::endl;

		std::vector<std::string> labels;
		cv::Mat img = cv::imread(image_file);
		PaddleDetection::image_slice(img, temp_save_path, 640, 0.25);
		vector<PaddleDetection::ObjectResult> result = single_circle_detect(temp_save_path, labels,device,category);
		bool is_rbox = false;
		// 图像可视化
		cv::Mat vis_img = PaddleDetection::VisualizeResult(img, result, labels, is_rbox);

		string output_path(output_dir);
		string filename = PaddleDetection::getFileName(image_file);
		
		output_path = output_path + "\\" + filename;
		cv::imwrite(output_path, vis_img);
		cout << "Visualized output saved as " << output_path.c_str() << endl;
		PaddleDetection::del_file(temp_save_path);

	}
	else if (result.size() == 1 && category == "strip")
	{
		std::cout << "Single Strip Detect" << std::endl;
		std::vector<std::string> labels;
		bool is_rbox = true;
		cv::Mat img = cv::imread(image_file);
		PaddleDetection::image_slice(img, temp_save_path, 640, 0.25);
		vector<PaddleDetection::ObjectResult> result = single_strip_detect(temp_save_path, labels,device,category);

		// 图像可视化
		cv::Mat vis_img = PaddleDetection::VisualizeResult(img, result, labels, is_rbox);

		string output_path(output_dir);

		string filename = PaddleDetection::getFileName(image_file);

		output_path = output_path + "\\" + filename;
		cv::imwrite(output_path, vis_img);
		cout << "Visualized output saved as " << output_path.c_str() << endl;
		PaddleDetection::del_file(temp_save_path);
	}
	else if (result.size() != 1 && category == "circle")
	{
		std::cout << "Multiple Circle Detect" << std::endl;
		std::vector<std::string> labels;
		vector<string> save_patch_path;
		vector<vector<int>> final_rect;
		vector<int> temp_rect;
		//temp_rect.reserve(20);
		for (int i = 0; i < result.size(); i++)
		{
			temp_rect.push_back(result[i].rect[0]);
			temp_rect.push_back(result[i].rect[1]);
			temp_rect.push_back(result[i].rect[2]);
			temp_rect.push_back(result[i].rect[3]);
			final_rect.push_back(temp_rect);
			temp_rect.clear();
		}
		vector<cv::Mat> patches = PaddleDetection::split_image(final_rect, image_file, 100, output_dir, save_patch_path);

		for (int i = 0; i < patches.size(); i++)
		{
			cv::Mat img = patches[i];
			PaddleDetection::image_slice(img, temp_save_path, 640, 0.25);
			vector<PaddleDetection::ObjectResult> result = multiple_circle_detect(temp_save_path, labels,device, category);
			bool is_rbox = false;
			// 图像可视化
			cv::Mat vis_img = PaddleDetection::VisualizeResult(img, result, labels, is_rbox);

			string output_path(output_dir);
			string filename = PaddleDetection::getFileName(save_patch_path[i]);

			output_path = output_path + "\\" + filename;
			cv::imwrite(output_path, vis_img);
			cout << "Visualized output saved as " << output_path.c_str() << endl;
			PaddleDetection::del_file(temp_save_path);
		}

	}
	else
	{
		std::cout << "Multiple Strip Detect" << std::endl;
		std::vector<std::string> labels;
		vector<string> save_patch_path;
		vector<vector<int>> final_rect;
		vector<int> temp_rect;
		//temp_rect.reserve(20);
		for (int i = 0; i < result.size(); i++)
		{
			temp_rect.push_back(result[i].rect[0]);
			temp_rect.push_back(result[i].rect[1]);
			temp_rect.push_back(result[i].rect[2]);
			temp_rect.push_back(result[i].rect[3]);
			final_rect.push_back(temp_rect);
			temp_rect.clear();
		}
		vector<cv::Mat> patches = PaddleDetection::split_image(final_rect, image_file, 100, output_dir, save_patch_path);

		for (int i = 0; i < patches.size(); i++)
		{
			cv::Mat img = patches[i];
			PaddleDetection::image_slice(img, temp_save_path, 640, 0.25);
			vector<PaddleDetection::ObjectResult> result = multiple_strip_detect(temp_save_path, labels,device,category);
			bool is_rbox = true;
			// 图像可视化
			cv::Mat vis_img = PaddleDetection::VisualizeResult(img, result, labels, is_rbox);

			string output_path(output_dir);
			string filename = PaddleDetection::getFileName(save_patch_path[i]);

			output_path = output_path + "\\" + filename;
			cv::imwrite(output_path, vis_img);
			cout << "Visualized output saved as " << output_path.c_str() << endl;
			PaddleDetection::del_file(temp_save_path);
		}
	}

	std::cout << "Finish" << std::endl;
	_rmdir(temp_save_path.c_str());
	return 0;

}