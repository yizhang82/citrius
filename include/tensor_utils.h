#pragma once

#include "exceptions.h"
#include "tensor.h"

#include <stdexcept>

#define ENSURE_TENSOR_DEFINED(tensor)                                               \
    do {                                                                            \
        if (!(tensor).defined()) {                                                   \
            throw std::invalid_argument(#tensor " must be defined");                \
        }                                                                            \
    } while (false)

#define ENSURE_TENSOR_SHAPE(tensor, expected_shape)                                 \
    do {                                                                             \
        if ((tensor).shape() != (expected_shape)) {                                  \
            throw std::invalid_argument(#tensor " has an unexpected shape");        \
        }                                                                            \
    } while (false)

#define ENSURE_TENSOR_DIM(tensor, expected_dim)                                     \
    do {                                                                             \
        if ((tensor).ndim() != static_cast<std::size_t>(expected_dim)) {             \
            throw std::invalid_argument(#tensor " has an unexpected number of dimensions"); \
        }                                                                            \
    } while (false)

#define ENSURE_TENSOR_DTYPE(tensor, expected_dtype)                                 \
    do {                                                                             \
        if ((tensor).dtype() != (expected_dtype)) {                                  \
            throw std::invalid_argument(#tensor " has an unexpected dtype");       \
        }                                                                            \
    } while (false)

#define ENSURE_TENSOR_DEVICE_MATCH_2(first, second)                                 \
    do {                                                                             \
        if ((first).device() != (second).device()) {                                 \
            throw citrius::DeviceMismatchException(                                 \
                #first " and " #second " must be on the same device");             \
        }                                                                            \
    } while (false)

#define ENSURE_TENSOR_DEVICE_MATCH_3(first, second, third)                          \
    do {                                                                             \
        if ((first).device() != (second).device() ||                                 \
            (first).device() != (third).device()) {                                  \
            throw citrius::DeviceMismatchException(                                 \
                #first ", " #second ", and " #third " must be on the same device"); \
        }                                                                            \
    } while (false)
