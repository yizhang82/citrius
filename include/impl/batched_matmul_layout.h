#pragma once

#include "tensor.h"
#include "tensor_utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace citrius::impl {

struct BatchedLayout {
    Shape output;
    std::vector<std::int64_t> a_offsets, b_offsets;
};
inline BatchedLayout make_batched_layout(const Tensor& a, const Tensor& b) {
    ENSURE_TENSOR_DEFINED(a);
    ENSURE_TENSOR_DEFINED(b);
    ENSURE_TENSOR_DTYPE(a, DType::Float32);
    ENSURE_TENSOR_DTYPE(b, DType::Float32);
    if (a.ndim() < 2 || b.ndim() < 2 || (a.ndim() == 2 && b.ndim() == 2))
        throw std::invalid_argument("batched_matmul expects at least one batched input");
    if (a.shape().back() != b.shape()[b.ndim() - 2])
        throw std::invalid_argument("batched_matmul inner dimensions must match");
    Shape as(a.shape().begin(), a.shape().end() - 2), bs(b.shape().begin(), b.shape().end() - 2);
    auto rank = std::max(as.size(), bs.size());
    Shape os(rank, 1);
    for (std::size_t x = 0; x < rank; ++x) {
        auto av = x < as.size() ? as[as.size() - 1 - x] : 1,
             bv = x < bs.size() ? bs[bs.size() - 1 - x] : 1;
        if (av != bv && av != 1 && bv != 1)
            throw std::invalid_argument("batched_matmul batch dimensions are not broadcastable");
        os[rank - 1 - x] = std::max(av, bv);
    }
    auto st = [](const Shape& s) {
        Strides r(s.size(), 1);
        for (std::size_t i = s.size(); i-- > 1;)
            r[i - 1] = r[i] * s[i];
        return r;
    };
    auto ost = st(os), ast = st(as), bst = st(bs);
    auto map = [&](std::int64_t q, const Shape& s, const Strides& t) {
        std::int64_t z = 0;
        auto p = os.size() - s.size();
        for (std::size_t i = p; i < os.size(); ++i) {
            auto c = (q / ost[i]) % os[i];
            if (s[i - p] != 1)
                z += c * t[i - p];
        }
        return z;
    };
    auto count = std::accumulate(os.begin(), os.end(), std::int64_t{1}, std::multiplies<>());
    BatchedLayout l;
    l.output = os;
    l.output.push_back(a.shape()[a.ndim() - 2]);
    l.output.push_back(b.shape().back());
    l.a_offsets.resize(count);
    l.b_offsets.resize(count);
    auto az = a.shape()[a.ndim() - 2] * a.shape().back(),
         bz = b.shape()[b.ndim() - 2] * b.shape().back();
    for (std::int64_t i = 0; i < count; ++i) {
        l.a_offsets[i] = map(i, as, ast) * az;
        l.b_offsets[i] = map(i, bs, bst) * bz;
    }
    return l;
}
} // namespace citrius::impl
