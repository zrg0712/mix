#pragma once
#include<string>

#ifdef DEPLOYMENTDLL_EXPORTS
#define DETECT_API __declspec(dllexport)
#else
#define DETECT_API __declspec(dllimport)
#endif // DEPLOYMENTDLL_EXPORTS


extern "C" DETECT_API int detectMain(std::string image_file, std::string device, std::string category);

