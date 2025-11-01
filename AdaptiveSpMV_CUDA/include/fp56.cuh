#ifndef FP56_H
#define FP56_H

#include <stdint.h>

struct fp56 {
    uint64_t bits : 56;

    __device__ fp56() : bits(0) {}
    __device__ fp56(float val) {
        uint64_t* p = reinterpret_cast<uint64_t*>(&val);
        bits = (*p >> 8) & 0xFFFFFFFFFFFFFF;
    }

    __device__ float to_float() const {
        uint64_t temp = (bits & 0xFFFFFFFFFFFFFF) << 8;
        float* p = reinterpret_cast<float*>(&temp);
        return *p;
    }

    // Arithmetic operations
    __device__ fp56 operator+(const fp56& other) const {
        return fp56(this->to_float() + other.to_float());
    }

    __device__ fp56 operator-(const fp56& other) const {
        return fp56(this->to_float() - other.to_float());
    }

    __device__ fp56 operator*(const fp56& other) const {
        return fp56(this->to_float() * other.to_float());
    }

    __device__ fp56 operator/(const fp56& other) const {
        return fp56(this->to_float() / other.to_float());
    }
};

#endif // FP56_H
