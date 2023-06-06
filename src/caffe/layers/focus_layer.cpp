#include "caffe/layers/focus_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void FocusLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
      CHECK_NE(top[0], bottom[0]) << this->type()
                                  << " Layer does not "
                                     "allow in-place computation.";
      FocusParameter focus_param = this->layer_param_.focus_param();
      CHECK_EQ(focus_param.has_stride(), true)
          << this->type() << " Layer needs stride param.";
      reverse_ = focus_param.reverse();
      stride_ = focus_param.stride();
      channels_ = bottom[0]->channels();
      height_ = bottom[0]->height();
      width_ = bottom[0]->width();
      batch_num_ = bottom[0]->num();

      diff_.Reshape(batch_num_, channels_, height_, width_);

      if (reverse_) {
        focused_channels_ = channels_ / (stride_ * stride_);
        focused_width_ = width_ * stride_;
        focused_height_ = height_ * stride_;
        } else {
            focused_channels_ = channels_ * stride_ * stride_;
            focused_height_ = height_ / stride_;
            focused_width_ = width_ / stride_;
            //std::cout << focused_channels_ << " " << focused_height_ << " " << focused_width_ << std::endl;
        }
    }

    template<typename Dtype>
    void FocusLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
        top[0]->Reshape(batch_num_, focused_channels_,
                        focused_height_, focused_width_);
    }

    template<typename Dtype>
    void FocusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        focus_cpu(bottom_data, width_, height_,
                  channels_, batch_num_, stride_, !reverse_, top_data);
    }

    template<typename Dtype>
    void FocusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
        if(!propagate_down[0]){
            return;
        }
        //const Dtype *top_diff = top[0]->cpu_diff();
        const Dtype *top_diff = diff_.mutable_cpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        focus_cpu(top_diff, width_, height_,
                  channels_, batch_num_, stride_, !reverse_, bottom_diff);
    }
#ifdef CPU_ONLY
STUB_GPU(FocusLayer);
#endif
    INSTANTIATE_CLASS(FocusLayer);

    REGISTER_LAYER_CLASS(Focus);

}  // namespace caffe