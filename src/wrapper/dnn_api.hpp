#ifndef __CAFFE_HEAER_H__
#define __CAFFE_HEAER_H__

#include <vector>
#include <string>

#include "ApiExport.h"

/* Data types */
// Handle to Caffe Net 
typedef void* NetHandle;

class DLL_EXPORTS Blob4F {
 public:
	int		dims[4];
	float*	data;
	bool	ownData;
	int num();
	int channels();
	int height();
	int width();
	int count();
	Blob4F();
	void reset();
	void create(int n,int ch,int h,int w);
	void createNoCopy(int n, int ch, int h, int w,float* sharedData);
	void destroy();
};

template<typename SrcType, typename DstType>
void ImageToTensor(SrcType* src, int height, int width, int channel,
	DstType* dst, int dims[4])
{
	// set dims
	dims[0] = 1;
	dims[1] = channel;
	dims[2] = height;
	dims[3] = width;
	int plane = height * width;
	int line = width;

	// convert data
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int c = 0; c < channel; c++)
			{
				dst[(channel - c - 1) * plane + y * line + x] = DstType(src[y * width * channel + x * channel + c]);
			}
		}
	}
}

template<typename SrcType, typename DstType>
void InterlacedToPlanar(SrcType* src, int height, int width, int channel,DstType* dst)
{
	int plane = height * width;
	int line = width;

	// convert data
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int c = 0; c < channel; c++)
			{
				dst[(channel - c - 1) * plane + y * line + x] = DstType(src[y * width * channel + x * channel + c]);
			}
		}
	}
}

template<typename SrcType, typename DstType>
void PlanarToInterlaced(SrcType* src, int height, int width, int channel, DstType* dst)
{
	int plane = height * width;
	int line = width;

	// convert data
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int c = 0; c < channel; c++)
			{
				dst[y * width * channel + x * channel + c] = DstType(src[(channel - c - 1) * plane + y * line + x]);
			}
		}
	}
}

template<typename DstType>
void Blob4FToVector(Blob4F& blob, std::vector<DstType>& vec)
{
	vec.resize(blob.count());
	for (int i = 0; i < blob.count(); i++)
	{
		vec[i] = blob.data[i];
	}
}

/* Caffe Model */
// @param useGpu 0 - CPU, 1 - GPU
// @param gpuId GPU mode gpu id, not used for CPU model.
DLLAPI(int)			caffe_set_mode(int useGpu, int gpuId);

/* Prediction APIs */
// Load a pretrained net
DLLAPI(NetHandle) 	caffe_net_load(const char* prototxt, const char* weights,int isPhaseTest=1);
// Load a net structure
DLLAPI(NetHandle) 	caffe_net_loadRaw(const char* prototxt, int isPhaseTest=1);
DLLAPI(void)        caffe_net_release(NetHandle handle);

// Set blob data for given blob name
bool	DLL_EXPORTS	caffe_net_setBlob(NetHandle net, const char* blobName, Blob4F blob);
// Get blob data for given blob name
Blob4F	DLL_EXPORTS	caffe_net_getBlob(NetHandle net, const char* blobName);
// Run forward 
DLLAPI(void) 		caffe_net_forward(NetHandle net);
DLLAPI(void)		caffe_net_forwardFromTo(NetHandle net, int from, int to);

//////////////////////////////////////////////////////////////////////////////////
/* Layer */
// Return all layer names
const std::vector<std::string> DLL_EXPORTS caffe_net_getLayerNames(NetHandle net);
// Return layer index
int		DLL_EXPORTS caffe_net_getLayerIndex(NetHandle net, const char* layerName);
// Return Input blob size
int     DLL_EXPORTS caffe_net_getInputBlobSize(NetHandle net);
// Return Input blob name
const std::string DLL_EXPORTS caffe_net_getInputBlobName(NetHandle net, int index);
//////////////////////////////////////////////////////////////////////////////////
// Compatible with old apis
typedef void* CAFFE_HANDLE;
DLLAPI(CAFFE_HANDLE)        caffe_create_handle(const char* model_path, const char* weights_path);
DLLAPI(void)                caffe_release_handle(CAFFE_HANDLE handle);
// Run the forward pass and get the blob data specified by 'blob_name'
std::vector<float> DLL_EXPORTS  caffe_extract_feature(CAFFE_HANDLE handle, const char* blob_name,
	unsigned char* bgr, int type, int width, int height);


#endif /* __CAFFE_HEAER_H__ */

