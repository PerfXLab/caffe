#include "caffe/caffe.hpp"

namespace caffe {

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template <typename T>
bool SortProposalPairDescend(const pair<T, T*>& pair1,
                             const pair<T, T*>& pair2);

}