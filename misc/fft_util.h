/*
 * Minimal radix-2 Cooley-Tukey FFT, pure C.
 * Complex numbers are represented as float[2] = { real, imag }.
 *
 * This file is part of mpv.
 * LGPL 2.1+
 */

#ifndef MP_FFT_UTIL_H
#define MP_FFT_UTIL_H

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float fft_amp(const float z[2])
{
    return logf(z[0] * z[0] + z[1] * z[1]);
}

/* In-place radix-2 iterative FFT.
 * in[n]       -> real input samples
 * out[n][2]   -> complex output (interleaved re,im)
 */
static inline void fft_compute(const float *in, float out[][2], size_t n)
{
    if (n == 0 || (n & (n - 1)) != 0)
        return;

    /* copy real input into complex array */
    for (size_t i = 0; i < n; ++i) {
        out[i][0] = in[i];
        out[i][1] = 0.0f;
    }

    /* bit-reversal permutation */
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float tmp0 = out[i][0], tmp1 = out[i][1];
            out[i][0] = out[j][0]; out[i][1] = out[j][1];
            out[j][0] = tmp0;      out[j][1] = tmp1;
        }
    }

    /* butterfly passes */
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wlen_re = cosf(angle);
        float wlen_im = sinf(angle);
        for (size_t i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (size_t k = 0; k < len / 2; ++k) {
                /* u = out[i+k] */
                float u_re = out[i + k][0];
                float u_im = out[i + k][1];
                /* v = out[i+k+len/2] * w */
                size_t idx = i + k + len / 2;
                float v_re = out[idx][0] * w_re - out[idx][1] * w_im;
                float v_im = out[idx][0] * w_im + out[idx][1] * w_re;
                out[i + k][0] = u_re + v_re;
                out[i + k][1] = u_im + v_im;
                out[idx][0]   = u_re - v_re;
                out[idx][1]   = u_im - v_im;
                /* w *= wlen */
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                float nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

#endif /* MP_FFT_UTIL_H */
