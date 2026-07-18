#include "nn/functional.h"

#include "operations.h"
#include "reduction_operations.h"

namespace citrius::nn::functional {

Tensor softmax(const Tensor& tensor, std::int64_t dim) {
    const Tensor shifted = sub(tensor, citrius::max(tensor, dim, true));
    const Tensor exponentials = citrius::exp(shifted);
    return div(exponentials, sum(exponentials, dim, true));
}

} // namespace citrius::nn::functional
