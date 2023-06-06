#ifndef CAFFE_FOCUS_LAYER_HPP_
#define CAFFE_FOCUS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
    template<typename Dtype>
    class FocusLayer : public Layer<Dtype> {
    public:
        explicit FocusLayer(const LayerParameter &param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const { return "Focus"; }

        virtual inline int ExactNumBottomBlobs() const { return 1; }

        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:


        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        int stride_;
        bool reverse_;
        int batch_num_;
        int channels_;
        int focused_channels_;
        int height_, width_;
        int focused_height_, focused_width_;
        Blob<Dtype> diff_;
    };
    template<typename Dtype>
    void focus_cpu(Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
    {
        int b,i,j,k;
        int out_c = c/(stride*stride);

        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h; ++j){
                    for(i = 0; i < w; ++i){
                        int in_index  = i + w*(j + h*(k + c*b));
                        int c2 = k % out_c;
                        int offset = k / out_c;
                        int w2 = i*stride + offset % stride;
                        int h2 = j*stride + offset / stride;
                        int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                        if(forward) out[out_index] = x[in_index];
                        else out[in_index] = x[out_index];
                    }
                }
            }
        }
    }

    template<typename Dtype>
    void focus_cpu(const Dtype *srcData, const int width, const int height,
                   const int channels, const int b_n, const int focusStride,
                   const bool forward, Dtype *dstData) {

	int outChannels = channels * focusStride * focusStride;
	int outHeight = height / focusStride;
	int outWidth = width / focusStride;

	for (int y = 0; y < outHeight; ++y) {
		for (int x = 0; x < outWidth; ++x) {
			for (int c = 0; c < outChannels; ++c) {
				int out_index = x + outWidth*(y + outHeight*c);

				int step = c / channels;
				int x_offset = step % focusStride;
				int y_offset = focusStride * ((step / focusStride) % focusStride);

				int in_x = x * focusStride + x_offset;
				
				int out_seq_y = y + c*outHeight;
				int in_intermediate_y = out_seq_y*2 - out_seq_y%2;
				in_intermediate_y = in_intermediate_y % (channels*height);
				int in_c = in_intermediate_y / height;
				int in_y = in_intermediate_y % height + y_offset;
						
				int in_index = in_x + width*(in_y + height*in_c);
				dstData[out_index] = srcData[in_index];
			}
		}
	}
    }



}  // namespace caffe

#endif  // CAFFE_FOCUS_LAYER_HPP_