#ifndef TTL_RANDOM_H
#define TTL_RANDOM_H
#include "core/types.h"

/*
*   Init that must be called in order to correctly generate random numbers.
*/
void randomInit(usize seed);

/*
*   Returns value sampled from uniform distribution in [0.0, 1.0]
*/
float uniform();

/*
*   Returns value sampled from uniform distribution in [low, high)
*/
float uniformRange(float low, float high);

/*
*   Returns value sampled from gaussian dsitribtuion with defined mean and std
*/
float gaussian(float mean, float std);
#endif