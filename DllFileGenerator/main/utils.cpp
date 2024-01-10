#include "utils.h"

using namespace std;

namespace PaddleDetection
{
	void nms(vector<ObjectResult>& input_boxes, float nms_threshold)
	{
		sort(input_boxes.begin(), input_boxes.end(), [](ObjectResult a, ObjectResult b) {return a.confidence > b.confidence; });

		vector<float> vArea(input_boxes.size());
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{

			vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) *
				(input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);

		}
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			for (int j = i + 1; j < int(input_boxes.size());)
			{
				float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
				float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
				float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
				float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
				float w = (std::max)(float(0), xx2 - xx1 + 1);
				float h = (std::max)(float(0), yy2 - yy1 + 1);
				float inter = w * h;
				//float ovr = inter / (vArea[i] + vArea[j] - inter);
				float ovr = inter / (std::min(vArea[i], vArea[j]));
				if (ovr >= nms_threshold) {
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else {
					j++;
				}
			}
		}
	}

	void nms_with_thread(vector<ObjectResult> input_boxes, vector<ObjectResult>& result)
	{
		//std::mutex my_mutex;
		//my_mutex.lock();
		sort(input_boxes.begin(), input_boxes.end(), [](ObjectResult a, ObjectResult b) {return a.confidence > b.confidence; });
		vector<float> vArea(input_boxes.size());
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) *
				(input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
		}
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			for (int j = i + 1; j < int(input_boxes.size());)
			{
				float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
				float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
				float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
				float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
				float w = (std::max)(float(0), xx2 - xx1 + 1);
				float h = (std::max)(float(0), yy2 - yy1 + 1);
				float inter = w * h;
				//float ovr = inter / (vArea[i] + vArea[j] - inter);
				float ovr = inter / std::min(vArea[i], vArea[j]);
				if (ovr >= 0.25) {
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else {
					j++;
				}
			}
		}
		//my_mutex.lock();
		result.insert(result.end(), input_boxes.begin(), input_boxes.end());
		//my_mutex.unlock();
	}

	vector<ObjectResult> nms_plus(vector<ObjectResult>& input_boxes, float nms_threshold)
	{
		vector<ObjectResult> result;
		vector<thread> my_threads;
		vector<vector<ObjectResult>> x_result;
		x_result.resize(15);
		int cut = input_boxes.size() / 10;
		for (int i = 0; i <= 10; i++)
		{

			std::vector<ObjectResult>::const_iterator first1;
			std::vector<ObjectResult>::const_iterator last1;
			if (i != 10)
			{
				first1 = input_boxes.begin() + cut * i;
				last1 = input_boxes.begin() + cut * (i + 1);
			}
			else
			{
				first1 = input_boxes.begin() + cut * i;
				last1 = input_boxes.end();
			}
			vector<ObjectResult> temp(first1, last1);

			my_threads.push_back(thread(PaddleDetection::nms_with_thread, temp, std::ref(x_result[i])));

		}

		for (auto it = my_threads.begin(); it != my_threads.end(); it++)
		{
			it->join();
		}

		for (int i = 0; i < x_result.size(); i++)
		{
			result.insert(result.end(), x_result[i].begin(), x_result[i].end());
		}

		return result;
	}

	void strip_nms(vector<ObjectResult>& input_boxes, float nms_threshold)
	{
		sort(input_boxes.begin(), input_boxes.end(), [](ObjectResult a, ObjectResult b) {return a.confidence > b.confidence; });
		vector<float> vArea(input_boxes.size());
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vector<cv::Point> contour;
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[0], input_boxes.at(i).rect[1]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[2], input_boxes.at(i).rect[3]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[4], input_boxes.at(i).rect[5]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[6], input_boxes.at(i).rect[7]));

			vArea[i] = cv::contourArea(contour);
			contour.clear();

		}
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vector<cv::Point> p;
			cv::Point p1(input_boxes.at(i).rect[0], input_boxes.at(i).rect[1]);
			cv::Point p2(input_boxes.at(i).rect[2], input_boxes.at(i).rect[3]);
			cv::Point p3(input_boxes.at(i).rect[4], input_boxes.at(i).rect[5]);
			cv::Point p4(input_boxes.at(i).rect[6], input_boxes.at(i).rect[7]);
			p = { p1,p2,p3,p4 };
			//cv::RotatedRect p(p1, p2, p3);
			auto areaRect_p = cv::minAreaRect(p);

			for (int j = i + 1; j < int(input_boxes.size());)
			{
				vector<cv::Point> v;
				cv::Point v1 = cv::Point2f(input_boxes.at(j).rect[0], input_boxes.at(j).rect[1]);
				cv::Point v2 = cv::Point2f(input_boxes.at(j).rect[2], input_boxes.at(j).rect[3]);
				cv::Point v3 = cv::Point2f(input_boxes.at(j).rect[4], input_boxes.at(j).rect[5]);
				cv::Point v4 = cv::Point2f(input_boxes.at(j).rect[6], input_boxes.at(j).rect[7]);
				v = { v1,v2,v3,v4 };
				auto areaRect_v = cv::minAreaRect(v);

				std::vector<cv::Point2f> outputArray;
				cv::rotatedRectangleIntersection(areaRect_p, areaRect_v, outputArray);

				// 获取相交点
				vector<cv::Point> inter_point;
				for (int k = 0; k < outputArray.size(); k++)
				{
					inter_point.push_back(outputArray[k]);
				}
				if (inter_point.size() != 0)
				{
					float inter = cv::contourArea(inter_point);
					inter_point.clear();
					//float ovr = inter / (vArea[i] + vArea[j] - inter);
					float ovr = inter / std::min(vArea[i], vArea[j]);
					if (ovr >= nms_threshold)
					{
						input_boxes.erase(input_boxes.begin() + j);
						vArea.erase(vArea.begin() + j);
					}
					else
					{
						j++;
					}
				}
				else
				{
					j++;
				}
			}
		}
	}

	void strip_nms_with_thread(vector<ObjectResult> input_boxes, vector<ObjectResult>& result)
	{
		sort(input_boxes.begin(), input_boxes.end(), [](ObjectResult a, ObjectResult b) {return a.confidence > b.confidence; });
		vector<float> vArea(input_boxes.size());
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vector<cv::Point> contour;
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[0], input_boxes.at(i).rect[1]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[2], input_boxes.at(i).rect[3]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[4], input_boxes.at(i).rect[5]));
			contour.push_back(cv::Point2f(input_boxes.at(i).rect[6], input_boxes.at(i).rect[7]));

			vArea[i] = cv::contourArea(contour);

		}
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vector<cv::Point> p;
			cv::Point p1(input_boxes.at(i).rect[0], input_boxes.at(i).rect[1]);
			cv::Point p2(input_boxes.at(i).rect[2], input_boxes.at(i).rect[3]);
			cv::Point p3(input_boxes.at(i).rect[4], input_boxes.at(i).rect[5]);
			cv::Point p4(input_boxes.at(i).rect[6], input_boxes.at(i).rect[7]);
			p = { p1,p2,p3,p4 };
			//cv::RotatedRect p(p1, p2, p3);
			auto areaRect_p = cv::minAreaRect(p);

			for (int j = i + 1; j < int(input_boxes.size());)
			{
				vector<cv::Point> v;
				cv::Point v1 = cv::Point2f(input_boxes.at(j).rect[0], input_boxes.at(j).rect[1]);
				cv::Point v2 = cv::Point2f(input_boxes.at(j).rect[2], input_boxes.at(j).rect[3]);
				cv::Point v3 = cv::Point2f(input_boxes.at(j).rect[4], input_boxes.at(j).rect[5]);
				cv::Point v4 = cv::Point2f(input_boxes.at(j).rect[6], input_boxes.at(j).rect[7]);
				v = { v1,v2,v3,v4 };
				auto areaRect_v = cv::minAreaRect(v);

				std::vector<cv::Point2f> outputArray;
				cv::rotatedRectangleIntersection(areaRect_p, areaRect_v, outputArray);

				// 获取相交点
				vector<cv::Point> inter_point;
				for (int k = 0; k < outputArray.size(); k++)
				{
					inter_point.push_back(outputArray[k]);
				}
				if (inter_point.size() != 0)
				{
					float inter = cv::contourArea(inter_point);
					inter_point.clear();
					//float ovr = inter / (vArea[i] + vArea[j] - inter);
					float ovr = inter / std::min(vArea[i], vArea[j]);
					if (ovr >= 0.25)
					{
						input_boxes.erase(input_boxes.begin() + j);
						vArea.erase(vArea.begin() + j);
					}
					else
					{
						j++;
					}
				}
				else
				{
					j++;
				}
			}
		}
		result.insert(result.end(), input_boxes.begin(), input_boxes.end());
	}

	vector<ObjectResult> strip_nms_plus(vector<ObjectResult>& input_boxes, float nms_threshold)
	{
		vector<ObjectResult> result;
		vector<thread> my_threads;
		vector<vector<ObjectResult>> x_result;
		x_result.resize(15);
		int cut = input_boxes.size() / 12;
		for (int i = 0; i <= 12; i++)
		{
			std::vector<ObjectResult>::const_iterator first1;
			std::vector<ObjectResult>::const_iterator last1;
			if (i != 12)
			{
				first1 = input_boxes.begin() + cut * i;
				last1 = input_boxes.begin() + cut * (i + 1);
			}
			else
			{
				first1 = input_boxes.begin() + cut * i;
				last1 = input_boxes.end();
			}
			vector<ObjectResult> temp(first1, last1);
			//x_result.push_back(temp);
			//my_threads.push_back(thread(PaddleDetection::strip_nms_with_thread, temp, std::ref(result)));
			my_threads.push_back(thread(PaddleDetection::strip_nms_with_thread, temp, std::ref(x_result[i])));

		}

		for (auto it = my_threads.begin(); it != my_threads.end(); it++)
		{
			it->join();
		}
		for (int i = 0; i < x_result.size(); i++)
		{
			result.insert(result.end(), x_result[i].begin(), x_result[i].end());
		}
		return result;
	}

	string getFileName(const string& path)
	{
		auto x = path.c_str();
		string::size_type iPos;
		if (strstr(path.c_str(), "\\"))
		{
			iPos = path.find_last_of('\\') + 1;
		}
		else
		{
			iPos = path.find_last_of('/') + 1;
		}

		string filename = path.substr(iPos, path.length() - iPos);

		return filename;
	}

	vector<cv::Mat> split_image(vector<vector<int>> rect, string img_path, int offset, string save_dir, vector<string>& save_patch_path)
	{
		string basename = getFileName(img_path);
		cv::Mat img = cv::imread(img_path);
		vector<cv::Mat> patches;
		for (int i = 0; i < rect.size(); i++)
		{
			int left_x1 = ((rect[i][0] - offset) < 0) ? 0 : (rect[i][0] - offset);
			int left_y1 = ((rect[i][1] - offset) < 0) ? 0 : (rect[i][1] - offset);
			int left_x2 = ((rect[i][2] + offset) < 3072) ? (rect[i][2] + offset) : 3072;
			int left_y2 = ((rect[i][3] + offset) < 3072) ? (rect[i][3] + offset) : 3072;

			cv::Mat img_copy, img_cut;
			cv::Rect rect_roi = cv::Rect(left_x1, left_y1, left_x2 - left_x1, left_y2 - left_y1);
			img.copyTo(img_copy);
			img_cut = img_copy(rect_roi);
			cv::resize(img_cut, img_cut, cv::Size(2400, 2400));
			patches.push_back(img_cut);
			string img_path = save_dir + "\\" + basename.substr(0, basename.length() - 5) + "__" + to_string(left_x1) + "_" + to_string(left_y1) + ".png";
			save_patch_path.push_back(img_path);
			cv::imwrite(img_path, img_cut);
		}

		return patches;
	}

	vector<vector<int>> get_slice_boxes(int height, int width, int img_size, float overlap_ratio)
	{
		vector<vector<int>> slice_bboxes;
		vector<int> boxes;
		int y_min = 0;
		int y_max = 0;
		int overlap = int(overlap_ratio * img_size);
		while (y_max<height)
		{
			int x_min = 0;
			int x_max = 0;
			y_max = y_min + img_size;
			while (x_max< width)
			{
				
				x_max = x_min + img_size;
				if (y_max > height || x_max > width)
				{
					int xmax = std::min(width, x_max);
					int ymax = std::min(height, y_max);
					int xmin = std::max(0, xmax - img_size);
					int ymin = std::max(0, ymax - img_size);
					boxes.push_back(xmin);
					boxes.push_back(ymin);
					boxes.push_back(xmax);
					boxes.push_back(ymax);
					slice_bboxes.push_back(boxes);
					boxes.clear();
				}
				else
				{
					boxes.push_back(x_min);
					boxes.push_back(y_min);
					boxes.push_back(x_max);
					boxes.push_back(y_max);
					slice_bboxes.push_back(boxes);
					boxes.clear();
				}
				x_min = x_max - overlap;
			}
			y_min = y_max - overlap;
		}
		return slice_bboxes;
	}

	void image_slice(cv::Mat img, string temp_save_path, int size, float overlap_ratio)
	{
		if (!path_exists(temp_save_path)) {
			mkDir(temp_save_path);
		}

		int width = img.cols;
		int height = img.rows;

		vector<vector<int>> slice_bboxes = get_slice_boxes(height, width, size, overlap_ratio);

		for (int i = 0; i < slice_bboxes.size(); i++)
		{
			int left_x = slice_bboxes[i][0];
			int left_y = slice_bboxes[i][1];
			int right_x = slice_bboxes[i][2];
			int right_y = slice_bboxes[i][3];

			cv::Mat img_copy, img_cut;
			cv::Rect rect_roi = cv::Rect(left_x, left_y, right_x - left_x, right_y - left_y);
			img.copyTo(img_copy);

			img_cut = img_copy(rect_roi);

			string img_path = temp_save_path + "\\" + to_string(left_x) + "_" + to_string(left_y) + ".png";

			cv::imwrite(img_path, img_cut);
		}
	}

	bool path_exists(const std::string& path)
	{
		struct _stat buffer;
		return (_stat(path.c_str(), &buffer) == 0);
	}

	void mkDir(const std::string& path) {
		if (path_exists(path)) return;
		int ret = 0;
		ret = _mkdir(path.c_str());
		if (ret != 0) {
			std::string path_error(path);
			path_error += "mkdir failed";
			throw std::runtime_error(path_error);
		}
	}

	vector<int> get_offset(string img_file_path)
	{
		vector<int> offset;
		string::size_type iPos;
		if (strstr(img_file_path.c_str(), "\\"))
		{
			iPos = img_file_path.find_last_of('\\') + 1;
		}
		else
		{
			iPos = img_file_path.find_last_of('/') + 1;
		}

		string filename = img_file_path.substr(iPos, img_file_path.length() - iPos);


		//2.获取不带后缀的文件名
		string name = filename.substr(0, filename.rfind("."));

		if (strstr(name.c_str(), "_"))
		{
			iPos = name.find_last_of("_");
		}

		string r = name.substr(iPos + 1, name.length());

		string l = name.substr(0, iPos);

		int l_x = atoi(l.c_str());
		int r_x = atoi(r.c_str());

		offset.push_back(l_x);
		offset.push_back(r_x);

		return offset;
	}

	void del_file(string dir)
	{
		std::vector<std::string> all_img_paths;
		std::vector<cv::String> cv_all_img_paths;
		cv::glob(dir, cv_all_img_paths);
		for (const auto& img_path : cv_all_img_paths)
		{
			string path = img_path;
			
			remove(path.c_str());
			
		}

	}
}  // namespace PaddleDetection