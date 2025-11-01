#ifndef FP40_H
#define FP40_H

#include <stdint.h>

struct fp40 {
    uint64_t bits : 40;

    __device__ fp40() : bits(0) {}
    __device__ fp40(float val) {
        uint64_t* p = reinterpret_cast<uint64_t*>(&val);
        bits = (*p >> 24) & 0xFFFFFFFFFF;
    }

    __device__ float to_float() const {
        uint64_t temp = (bits & 0xFFFFFFFFFF) << 24;
        float* p = reinterpret_cast<float*>(&temp);
        return *p;
    }

    // Arithmetic operations
    __device__ fp40 operator+(const fp40& other) const {
        return fp40(this->to_float() + other.to_float());
    }

    __device__ fp40 operator-(const fp40& other) const {
        return fp40(this->to_float() - other.to_float());
    }

    __device__ fp40 operator*(const fp40& other) const {
        return fp40(this->to_float() * other.to_float());
    }

    __device__ fp40 operator/(const fp40& other) const {
        return fp40(this->to_float() / other.to_float());
    }
};

#endif // FP40_H
