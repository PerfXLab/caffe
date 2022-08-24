// ------------------------------------------------------------------
// Project: Mask R-CNN
// File: ROIAlignLayer
// Adopted from roi_pooling_layer.cpp (written by Ross Grischik)
// Author: Jasjeet Dhaliwal
// ------------------------------------------------------------------

#include "caffe/layers/roi_align_layer.hpp"

using std::ceil;
using std::fabs;
using std::floor;
using std::max;
using std::min;
using std::sqrt;
using std::vector;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0) << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0) << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h();
  pooled_width_ = roi_align_param.pooled_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  aligned_ = roi_align_param.aligned();
  topk_ = roi_align_param.topk();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  channels_ = bottom[2]->channels();
  //    height_ = bottom[0]->height();
  //    width_ = bottom[0]->width();
  top[0]->Reshape(topk_, channels_, pooled_height_, pooled_width_);
  vector<int> top_shape;
  top_shape.push_back(topk_);
  top_shape.push_back(bottom[0]->channels());
  top[1]->Reshape(top_shape);
  int shape_init[] = {topk_, channels_, pooled_height_, pooled_width_, 4};
  const vector<int> shape(shape_init,
                          shape_init + sizeof(shape_init) / sizeof(int));
  max_mult_.Reshape(shape);
  max_pts_.Reshape(shape);
}

// author: houxin
// date: 20220518
// reference:
// https://github.com/open-mmlab/mmcv/blob/58e32423f04048980946693772fb059d554eacd0/mmcv/ops/csrc/pytorch/cpu/roi_align.cpp#L384
//            https://github.com/jwyang/faster-rcnn.pytorch/blob/297e8e0f414636c47a6b13131052c9274cd38702/lib/model/csrc/cpu/ROIAlign_cpu.cpp#L162
template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  LOG(INFO) << "DOING CPU FORWARD NOW ";
  // proposals
  const Dtype *bottom_proposals = bottom[0]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[0]->num();
  // scores
  const Dtype *bottom_scores = bottom[1]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *top_rois = top[1]->mutable_cpu_data();
  // features, num = 4
  vector<Blob<Dtype> *> features;
  const int num = bottom.size();  // 6
  for (int i = 2; i < num; ++i) {
    features.push_back(bottom[i]);
  }
  // Retrieve all proposals and scores
  vector<pair<Dtype, Dtype *> > proposal_pairs;
  for (int i = 0; i < num_rois; ++i) {
    proposal_pairs.push_back(std::make_pair(
        bottom_scores[i], const_cast<Dtype *>(bottom_proposals)));
    bottom_proposals += bottom[0]->offset(1);
  }

  // Keep top k proposal per imagee
  std::sort(proposal_pairs.begin(), proposal_pairs.end(),
            SortProposalPairDescend<Dtype>);
  proposal_pairs.resize(topk_);

  // i + bottom_scores[i], formally bottom_scores[i]. i indicates the location
  // of every proposal
  for (int i = 0; i < topk_; i++) {
    proposal_pairs[i].first += i;
  }

  int top_rois_index = 0;
  typename vector<pair<Dtype, Dtype *> >::iterator itxx;
  for (itxx = proposal_pairs.begin(); itxx != proposal_pairs.end(); itxx++) {
    const Dtype *proposal = (*itxx).second;
    for (int roi = 0; roi < bottom[0]->channels(); ++roi) {
      top_rois[top_rois_index++] = proposal[roi];
    }
  }

  // roialign
  // step 1: match feature and proposal
  // step 2: compute
  int finest_scale = 56;
  Dtype offset = aligned_ ? (Dtype)0.5 : (Dtype)0.0;
  for (int i = 0; i < num - 2; ++i) {
    const Blob<Dtype> *feature = features[i];
    height_ = feature->height();
    width_ = feature->width();
    typename vector<pair<Dtype, Dtype *> >::iterator it;
    for (it = proposal_pairs.begin(); it != proposal_pairs.end();) {
      const Dtype *proposal = (*it).second;
      float scale =
          sqrt((proposal[3] - proposal[1]) * (proposal[4] - proposal[2]));
      bool filter = false;
      switch (i) {
        case 0:
          (scale < 2 * finest_scale) ? (filter = true) : (filter = false);
          break;
        case 1:
          ((scale >= 2 * finest_scale) and (scale < 4 * finest_scale))
              ? (filter = true)
              : (filter = false);
          break;
        case 2:
          ((scale >= 4 * finest_scale) and (scale < 8 * finest_scale))
              ? (filter = true)
              : (filter = false);
          break;
        case 3:
          (scale >= finest_scale * 8) ? (filter = true) : (filter = false);
          break;
      }  // match feature
      if (filter) {
        int location = (int)((*it).first);
        top_data = top[0]->mutable_cpu_data() + top[0]->offset(1) * location;

        it = proposal_pairs.erase(it);
        int batch_size = feature->num();

        // step2: compute
        int roi_batch_ind = proposal[0];
        Dtype roi_start_w = proposal[1] * spatial_scale_ - offset;
        Dtype roi_start_h = proposal[2] * spatial_scale_ - offset;
        Dtype roi_end_w = proposal[3] * spatial_scale_ - offset;
        Dtype roi_end_h = proposal[4] * spatial_scale_ - offset;

        CHECK_GE(roi_batch_ind, 0);
        CHECK_LT(roi_batch_ind, batch_size);
        // Util Values
        Dtype roi_height = roi_end_h - roi_start_h;
        Dtype roi_width = roi_end_w - roi_start_w;
        Dtype one = 1.0;
        Dtype zero = 0.0;
        if (aligned_) {
          roi_height = max(roi_height, zero);
          roi_width = max(roi_width, zero);
        } else {
          roi_height = max(roi_height, one);
          roi_width = max(roi_width, one);
        }

        const Dtype bin_size_h =
            static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
        const Dtype bin_size_w =
            static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);
        const Dtype *batch_data =
            feature->cpu_data() + feature->offset(roi_batch_ind);

        int roi_bin_grid_h = ceil(roi_height / pooled_height_);  // e.g., = 2
        int roi_bin_grid_w = ceil(roi_width / pooled_width_);

        const Dtype count =
            max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g., = 4;

        // core code
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              Dtype avgvalue = 0.0;
              const int pool_index = ph * pooled_width_ + pw;

              for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                const Dtype yy = roi_start_h + ph * bin_size_h +
                                 static_cast<Dtype>(iy + .5f) * bin_size_h /
                                     static_cast<Dtype>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                  const Dtype xx = roi_start_w + pw * bin_size_w +
                                   static_cast<Dtype>(ix + .5f) * bin_size_w /
                                       static_cast<Dtype>(roi_bin_grid_w);

                  Dtype x = xx;
                  Dtype y = yy;
                  if (y < -1.0 || y > height_ || x < -1.0 || x > width_) {
                    avgvalue += 0.0;
                    continue;
                  }
                  if (y <= 0) {
                    y = 0;
                  }
                  if (x <= 0) {
                    x = 0;
                  }

                  int y_low = (int)y;
                  int x_low = (int)x;
                  int y_high;
                  int x_high;

                  if (y_low >= height_ - 1) {
                    y_high = y_low = height_ - 1;
                    y = (Dtype)y_low;
                  } else {
                    y_high = y_low + 1;
                  }

                  if (x_low >= width_ - 1) {
                    x_high = x_low = width_ - 1;
                    x = (Dtype)x_low;
                  } else {
                    x_high = x_low + 1;
                  }

                  Dtype ly = y - y_low;
                  Dtype lx = x - x_low;
                  Dtype hy = 1. - ly, hx = 1. - lx;
                  Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                  int pos1 = y_low * width_ + x_low;
                  int pos2 = y_low * width_ + x_high;
                  int pos3 = y_high * width_ + x_low;
                  int pos4 = y_high * width_ + x_high;
                  avgvalue += batch_data[pos1] * w1 + batch_data[pos2] * w2 +
                              batch_data[pos3] * w3 + batch_data[pos4] * w4;

                }  // ix
              }    // iy

              avgvalue = avgvalue / count;
              top_data[pool_index] = avgvalue;
            }  // pw
          }    // ph
          // Increment all data pointers by one channel
          batch_data += features[i]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }  // channels
      } else {
        it++;
      }
    }  // proposal_pairs
    spatial_scale_ /= 2.0;
  }  // feature
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                        const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {
  const Dtype *bottom_rois = bottom[1]->cpu_data();
  const Dtype *top_diff = top[0]->cpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);

  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  const int *argmax_idx = max_pts_.cpu_data();
  const Dtype *argmax_mult = max_mult_.cpu_data();

  int index = 0;  // Current index
                  //  std::cout <<"Batch = " << batch_size << "\n";
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          index = (((((b * channels_) + c) * height_) + h) * width_) + w;
          // Go over every ROI
          Dtype gradient = 0.0;
          for (int n = 0; n < num_rois; ++n) {
            const Dtype *offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            CHECK_GE(roi_batch_ind, 0);
            CHECK_LT(roi_batch_ind, batch_size);

            int offset = (n * channels_ + c) * pooled_height_ * pooled_width_;
            int argmax_offset = offset * 4;
            const Dtype *offset_top_diff = top_diff + offset;
            const int *offset_argmax_idx = argmax_idx + argmax_offset;
            const Dtype *offset_argmax_mult = argmax_mult + argmax_offset;
            Dtype multiplier = 0.0;
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                for (int k = 0; k < 4; ++k) {
                  if (offset_argmax_idx[((ph * pooled_width_ + pw) * 4) + k] ==
                      index) {
                    multiplier =
                        offset_argmax_mult[((ph * pooled_width_ + pw) * 4) + k];
                    gradient +=
                        offset_top_diff[ph * pooled_width_ + pw] * multiplier;
                  }
                }
              }  // Pw
            }    // Ph
          }      // rois
          bottom_diff[index] = gradient;
        }  // width
      }    // height
    }      // channels
  }        // count
}

#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
