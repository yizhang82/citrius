#pragma once

#include <stdexcept>

namespace citrius {

class CitriusException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DeviceMismatchException : public CitriusException {
public:
    using CitriusException::CitriusException;
};

} // namespace citrius
