#ifndef FP48_H
#define FP48_H

#include <stdint.h>

struct fp48 {
    uint64_t bits : 48;

    __device__ fp48() : bits(0) {}
    __device__ fp48(float val) {
        uint64_t* p = reinterpret_cast<uint64_t*>(&val);
        bits = (*p >> 16) & 0xFFFFFFFFFFFF;
    }

    __device__ float to_float() const {
        uint64_t temp = (bits & 0xFFFFFFFFFFFF) << 16;
        float* p = reinterpret_cast<float*>(&temp);
        return *p;
    }

    // Arithmetic operations
    __device__ fp48 operator+(const fp48& other) const {
        return fp48(this->to_float() + other.to_float());
    }

    __device__ fp48 operator-(const fp48& other) const {
        return fp48(this->to_float() - other.to_float());
    }

    __device__ fp48 operator*(const fp48& other) const {
        return fp48(this->to_float() * other.to_float());
    }

    __device__ fp48 operator/(const fp48& other) const {
        return fp48(this->to_float() / other.to_float());
    }
};

#endif // FP48_H
