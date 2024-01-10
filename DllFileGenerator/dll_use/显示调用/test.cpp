#include <Windows.h>
#include <string>
using namespace std;
typedef int(*Dllfun)(string, string, string);
int main()
{
	Dllfun detect;
	HINSTANCE hdll;
	hdll = LoadLibrary("deploymentDLL.dll");
	if (hdll == NULL)  // 判断是否读取dll文件成功
	{
		return -1;
	}
	else
	{
		detect = (Dllfun)GetProcAddress(hdll, "detectMain");  // 对外开放的接口
		if (detect != NULL)
		{
			// 函数接收的参数 （待测图像所在路径，GPU/CPU，strip/circle）
			int num = detect("circle__1435_0_0_0.tiff", "GPU", "circle"); 
		}
	}

	return 0;
}