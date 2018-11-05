#include "dnn_api.hpp"
#include <algorithm>
#include <string>
#include <vector>

#include "androidlog.h"
#include "net.h"

// Blob4F
int Blob4F::num()
{
	return dims[0];
}
int Blob4F::channels()
{
	return dims[1];
}
int Blob4F::height()
{
	return dims[2];
}

int Blob4F::count()
{
	return dims[0] * dims[1] * dims[2] * dims[3];
}

int Blob4F::width()
{
	return dims[3];
}

Blob4F::Blob4F()
{
	ownData = false;
	reset();
}

void Blob4F::reset()
{
	dims[0] = dims[1] = dims[2] = dims[3] = 0;
	data = NULL;
}

void Blob4F::create(int n,int ch,int h,int w)
{
	dims[0] = n;
	dims[1] = ch;
	dims[2] = h;
	dims[3] = w;
	data = new float[n * ch * w * h];
	ownData = true;
}

void Blob4F::createNoCopy(int n, int ch, int h, int w, float* sharedData)
{
	dims[0] = n;
	dims[1] = ch;
	dims[2] = h;
	dims[3] = w;
	data = sharedData;
	ownData = false;
}

void Blob4F::destroy()
{
	if(ownData && data)
	{
		delete data;
		data = NULL;
	}
	reset();
}

class NetEx : public ncnn::Net
{
public:
	int get_blob_index(const char* name) const
	{
		return find_blob_index_by_name(name);
	}
	int get_layer_index(const char* name) const
	{
		return find_layer_index_by_name(name);
	}
	const std::vector<std::string> get_layer_names()
	{
		std::vector<std::string> names(layers.size());
		for (int i = 0; i < layers.size(); i++)
		{
			names[i] = layers[i]->name;
		}
		return names;
	}
};
class NetContext
{
public:
	NetEx net;
	ncnn::Extractor ex;
	int ex_state;   // 0 invalid 1 valid
	int data_state; // 0 feed 1 get
	NetContext()
	{
		ex_state = 0;
		data_state = 0;
	}
	~NetContext()
	{
		net.clear();
	}

	int load_param(const char* param, const char* bin)
	{
		int ret = 0;
		ret += net.load_param(param);
		ret += net.load_model(bin);
		return ret;
	}

	bool set_blob(const char* blobName, Blob4F blob)
	{
		// reset extractor
		if (1 == ex_state && 1 == data_state )
		{
			ex_state = 0;
		}
		// create new
		if (0 == ex_state)
		{
			ex = net.create_extractor();
			ex.reset();
			ex.set_light_mode(true);
			ex.set_num_threads(1);
			ex_state = 1;
			LOGE("Reset extractor");
		}
		data_state = 0;
		// feed
		ncnn::Mat flatten(blob.width()* blob.height() * blob.channels(), blob.data);
		ncnn::Mat input = flatten.reshape(blob.width(), blob.height() , blob.channels());
		return !ex.input(blobName, input);
	}

	Blob4F get_blob(const char* blobName)
	{
		Blob4F blob;
		if (0 == ex_state)
		{
			LOGE("Set input before extract blob:%s", blobName);
			return blob;
		}
		ncnn::Mat out;
		int err = ex.extract(blobName, out);
		data_state = 1;
		if (err)
		{
			LOGE("Get blob:%s error:%d", blobName, err);
			return blob;
		}
		blob.create(1, out.c, out.h, out.w);
		int count = out.c*out.h*out.w;
		ncnn::Mat flatten = out.reshape(count);
		memcpy(blob.data, flatten.data, count * sizeof(float));
		return blob;
	}
};

int	caffe_set_mode(int isGpu, int gpuId)
{
	LOGE("set mode not support");
	return 0;
}

/* Prediction APIs */
// Load a pretrained net
NetHandle 	caffe_net_load(const char* prototxt, const char* weights, int isPhaseTest )
{
	NetContext* net = new NetContext();
	int err = net->load_param(prototxt, weights);
	if (err)
	{
		delete net;
		return NULL;
	}
	return (NetHandle)net;
}

// Load a net structure
NetHandle caffe_net_loadRaw(const char* prototxt, int isPhaseTest )
{
	NetContext* net = new NetContext();
	int err = net->net.load_param(prototxt);
	if (err)
	{
		delete net;
		return NULL;
	}
	return (NetHandle)net;
}

void caffe_net_release(NetHandle net)
{
	NetContext* pNet = (NetContext*)net;
	if (!pNet)
	{
		return ;
	}

	delete pNet;
}

// Set blob data for given blob name
bool caffe_net_setBlob(NetHandle net, const char* blobName, Blob4F blob)
{
	NetContext* pNet = (NetContext*)net;
	if (!pNet)
	{
		return false;
	}
	int err = pNet->set_blob(blobName, blob);
	return !err;
}

// Get blob data for given blob name
Blob4F caffe_net_getBlob(NetHandle net, const char* blobName)
{
	NetContext* pNet = (NetContext*)net;
	if (!pNet)
	{
		return Blob4F();
	}
	return pNet->get_blob(blobName);
}

// Run forward 
void caffe_net_forward(NetHandle net)
{
}
	
void caffe_net_forwardFromTo(NetHandle net, int from, int to)
{
}

const std::vector<std::string> caffe_net_getLayerNames(NetHandle net)
{
	std::vector<std::string> names;
	NetContext* pNet = (NetContext*)net;
	if (!pNet)
	{
		return names;
	}
	return pNet->net.get_layer_names();
}

int caffe_net_getLayerIndex(NetHandle net, const char* layerName)
{
	std::vector<std::string> names;
	NetContext* pNet = (NetContext*)net;
	if (!pNet)
	{
		return -1;
	}
	return pNet->net.get_layer_index(layerName);
}

int caffe_net_getInputBlobSize(NetHandle net)
{
	return 1;
}

const std::string caffe_net_getInputBlobName(NetHandle net, int index)
{
	return std::string("data");
}

//////////////////////////////////////////////////////////////////////////////////
// Compatible with old apis
CAFFE_HANDLE caffe_create_handle(const char* model_path, const char* weights_path)
{
	return caffe_net_load(model_path,weights_path);
}

void caffe_release_handle(CAFFE_HANDLE handle)
{
	caffe_net_release(handle);
}


/****************************************************************************************\
*                                  Matrix type (CvMat)                                   *
\****************************************************************************************/

#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_USRTYPE1 7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))

#define CV_AUTO_STEP  0x7fffffff
#define CV_WHOLE_ARR  cvSlice( 0, 0x3fffffff )

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)
#define CV_MAT_CONT_FLAG_SHIFT  14
#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)
#define CV_IS_MAT_CONT(flags)   ((flags) & CV_MAT_CONT_FLAG)
#define CV_IS_CONT_MAT          CV_IS_MAT_CONT
#define CV_SUBMAT_FLAG_SHIFT    15
#define CV_SUBMAT_FLAG          (1 << CV_SUBMAT_FLAG_SHIFT)
#define CV_IS_SUBMAT(flags)     ((flags) & CV_MAT_SUBMAT_FLAG)

#define CV_MAGIC_MASK       0xFFFF0000
#define CV_MAT_MAGIC_VAL    0x42420000
#define CV_TYPE_NAME_MAT    "opencv-matrix"

std::vector<float>  caffe_extract_feature(CAFFE_HANDLE handle, const char* blob_name,
	unsigned char* bgr, int type, int width, int height)
{
	std::vector<float> ret;
	int channels = CV_MAT_CN(type);
	int depth = CV_MAT_DEPTH(type);
	// only surport uchar and float input
	if (CV_8U != depth && CV_32F != depth)
	{
		return ret;
	}

	if (!handle)
	{
		return ret;
	}

	// fill blob
	Blob4F inputBlob;
	inputBlob.create(1, channels,height,width);
	switch (depth)
	{
	case CV_8U:
		ImageToTensor<unsigned char, float>(bgr, height,width, channels,inputBlob.data,inputBlob.dims);
		break;
	case CV_32F:
		ImageToTensor<float, float>((float*)bgr, height,width, channels, inputBlob.data, inputBlob.dims);
		break;
	}

	// feed to net
	caffe_net_setBlob(handle, "data", inputBlob);
	// run forward
	caffe_net_forward(handle);
	// extract result
	Blob4F featBlob = caffe_net_getBlob(handle, blob_name);
	Blob4FToVector<float>(featBlob, ret);
	featBlob.destroy();
	inputBlob.destroy();
	return ret;
}

