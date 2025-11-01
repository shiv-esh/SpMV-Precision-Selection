#ifndef FP24_H
#define FP24_H

#include <stdint.h>

struct fp24 {
    uint32_t bits : 24;

    __device__ fp24() : bits(0) {}
    __device__ fp24(float val) {
        uint32_t* p = reinterpret_cast<uint32_t*>(&val);
        bits = (*p >> 8) & 0xFFFFFF;
    }

    __device__ float to_float() const {
        uint32_t temp = (bits & 0xFFFFFF) << 8;
        float* p = reinterpret_cast<float*>(&temp);
        return *p;
    }

    // Arithmetic operations
    __device__ fp24 operator+(const fp24& other) const {
        return fp24(this->to_float() + other.to_float());
    }

    __device__ fp24 operator-(const fp24& other) const {
        return fp24(this->to_float() - other.to_float());
    }

    __device__ fp24 operator*(const fp24& other) const {
        return fp24(this->to_float() * other.to_float());
    }

    __device__ fp24 operator/(const fp24& other) const {
        return fp24(this->to_float() / other.to_float());
    }
};

#endif