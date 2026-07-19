#pragma once

#include "tensor.h"

#include <stdexcept>

#define CITRIUS_ENSURE_TENSOR_DEFINED(tensor)                                      \
    do {                                                                            \
        if (!(tensor).defined()) {                                                   \
            throw std::invalid_argument(#tensor " must be defined");                \
        }                                                                            \
    } while (false)

#define CITRIUS_ENSURE_TENSOR_SHAPE(tensor, expected_shape)                         \
    do {                                                                             \
        if ((tensor).shape() != (expected_shape)) {                                  \
            throw std::invalid_argument(#tensor " has an unexpected shape");        \
        }                                                                            \
    } while (false)

#define CITRIUS_ENSURE_TENSOR_DIM(tensor, expected_dim)                             \
    do {                                                                             \
        if ((tensor).ndim() != static_cast<std::size_t>(expected_dim)) {             \
            throw std::invalid_argument(#tensor " has an unexpected number of dimensions"); \
        }                                                                            \
    } while (false)
