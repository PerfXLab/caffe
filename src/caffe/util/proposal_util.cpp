#include "caffe/util/proposal_util.hpp"

namespace caffe {

template <typename T>
bool SortProposalPairDescend(const pair<T, T*>& pair1,
                             const pair<T, T*>& pair2)
{
    return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortProposalPairDescend(const pair<float, float*>& pair1,
                                      const pair<float, float*>& pair2);

template bool SortProposalPairDescend(const pair<double, double*>& pair1,
                                      const pair<double, double*>& pair2);

}