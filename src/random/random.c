#include <math.h>
#include "random/random.h"
#define M_PI 3.14159265358979323846

static usize state = 0;
static usize inc = 1u;
static usize seed = 0;

usize32 next() {
    usize oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    usize32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    usize32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void randomInit(usize s) {
    state = 0;
    inc = 1u;
    seed = s;

    next();
    state += seed;
    next();
}

float uniform() {
    usize32 bits;

    do {
        bits = next();
    } while (bits == 0);

    return bits * (1.0f / 4294967296.0f);
}

float uniformRange(float low, float high) {
    return low + uniform() * (high - low);
}

float gaussian(float mean, float std) {
    float u1 = uniform();
    float u2 = uniform();
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + z * std;
}