#include <Windows.h>
#include <string>
using namespace std;
typedef int(*Dllfun)(string, string, string);
int main()
{
	Dllfun detect;
	HINSTANCE hdll;
	hdll = LoadLibrary("deploymentDLL.dll");
	if (hdll == NULL)  // �ж��Ƿ��ȡdll�ļ��ɹ�
	{
		return -1;
	}
	else
	{
		detect = (Dllfun)GetProcAddress(hdll, "detectMain");  // ���⿪�ŵĽӿ�
		if (detect != NULL)
		{
			// �������յĲ��� ������ͼ������·����GPU/CPU��strip/circle��
			int num = detect("circle__1435_0_0_0.tiff", "GPU", "circle"); 
		}
	}

	return 0;
}