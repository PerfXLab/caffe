#include <cmath>
#include <vector>

#include "caffe/layers/yolov3_detection_output_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"
namespace caffe {
template <typename Dtype>
void Yolov3DetectionOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int len = 4 + num_class_ + 1;
  if(gaussian_box_) 
    len = 8 + num_class_ + 1;
  int mask_offset = 0;
  
  predicts_.clear();
  Dtype *class_score = new Dtype[num_class_];
  
  for (int t = 0; t < bottom.size(); t++) {
    side_w_ = bottom[t]->width();
    side_h_ = bottom[t]->height();
    std::cout<<"t = "<<side_w_<<"   "<<side_h_<<std::endl;
    int stride = side_w_*side_h_;
    swap_.ReshapeLike(*bottom[t]);
    Dtype* swap_data = swap_.mutable_gpu_data();
    const Dtype* input_data = bottom[t]->gpu_data();
    int nw = side_w_*anchors_scale_[t];
    int nh = side_h_*anchors_scale_[t];
    for (int b = 0; b < bottom[t]->num(); b++) {
      for (int n = 0; n < num_; ++n) {
        int index = n*len*stride  + b*bottom[0]->count(1);
        if(gaussian_box_) {
          caffe_gpu_logistic_activate(4 * side_w_*side_h_,input_data + index,swap_data +index );
          index = n*len*stride  + b*bottom[0]->count(1) + 4 * stride;
          caffe_copy(side_w_*side_h_, input_data + index, swap_data + index);		  
          index = n*len*stride  + b*bottom[0]->count(1) + 5 * stride;
          caffe_gpu_logistic_activate(side_w_*side_h_,input_data + index,swap_data +index );		  
          index = n*len*stride  + b*bottom[0]->count(1) + 6 * stride;
          caffe_copy(side_w_*side_h_, input_data + index, swap_data + index);
          index = n*len*stride  + b*bottom[0]->count(1) + 7 * stride;
          caffe_gpu_logistic_activate(side_w_*side_h_,input_data + index,swap_data +index );		  
          index = n*len*stride  + b*bottom[0]->count(1) + 8 * stride;
          caffe_gpu_logistic_activate((num_class_+1) * side_w_*side_h_,input_data + index,swap_data +index );
        }
        else {
          caffe_gpu_logistic_activate(2 * side_w_*side_h_,input_data + index,swap_data +index );
          index = n*len*stride  + b*bottom[0]->count(1) + 2 * stride;
          caffe_copy(2 * side_w_*side_h_, input_data + index, swap_data + index);
          index = n*len*stride  + b*bottom[0]->count(1) + 4 * stride;
          caffe_gpu_logistic_activate((num_class_+1) * side_w_*side_h_,input_data + index,swap_data +index );
        }
      }
      Dtype* swap_data = swap_.mutable_cpu_data();
      for (int s = 0; s < side_w_*side_h_; s++) {				
        for (int n = 0; n < num_; n++) {
          //LOG(INFO) << bottom[t]->count(1);
          int index = n*len*stride + s + b*bottom[t]->count(1);
          vector<Dtype> pred;
          Dtype* swap_data = swap_.mutable_cpu_data();
          if(gaussian_box_) {
            for (int c = 9; c < len; ++c) {
              int index2 = c*stride + index;
              class_score[c - 9] = (swap_data[index2 + 0]);
            }
          }
          else {
            for (int c = 5; c < len; ++c) {
              int index2 = c*stride + index;
               class_score[c - 5] = (swap_data[index2 + 0]);
            }
          }
          int y2 = s / side_w_;
          int x2 = s % side_w_;
          Dtype obj_score;
          if(gaussian_box_) {
            Dtype uc_ver = 4.0 - swap_data[index + 1 * stride] - swap_data[index + 1 * stride] - swap_data[index + 3 * stride] - swap_data[index + 5 * stride] - swap_data[index + 7 * stride];             
            obj_score = swap_data[index + 8 * stride] * uc_ver/4.0;
          }
          else {
            obj_score = swap_data[index + 4 * stride];
          }
          PredictionResult<Dtype> predict;
          for (int c = 0; c < num_class_; ++c) {
            class_score[c] *= obj_score;
            if (class_score[c] > confidence_threshold_)
            {						
              if(gaussian_box_) {
                get_gaussian_yolo_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
              }
              else {
                get_region_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
              }
              predict.x = pred[0];
              predict.y = pred[1];
              predict.w = pred[2];
              predict.h = pred[3];
              predict.classType = c ;
              predict.confidence = class_score[c];
              correct_yolo_boxes(predict,side_w_,side_h_,nw,nh,1);
              predicts_.push_back(predict);							

              //LOG(INFO) << predict.x << "," << predict.y << "," << predict.w << "," << predict.h;
              //LOG(INFO) << predict.confidence;
            }
          }
        }
      }
    }
    mask_offset += groups_num_;
    
  }

  delete[] class_score;

  Forward_cpu(bottom,top);	
}

template <typename Dtype>
void Yolov3DetectionOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}

INSTANTIATE_LAYER_GPU_FUNCS(Yolov3DetectionOutputLayer);


}  // namespace caffe
