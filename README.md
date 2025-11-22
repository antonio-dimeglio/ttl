# Tiny Tensor Library (TTL)

TTL is designed to be a lightweight and easy to use library for tensors. 
This is achieved by supporting only one datatype, F32.
The library tries to support _most_ of what numpy APIs, with the goal to use this library as the backend for a full python library in its v2. 

# Dependencies
TTL tries to be reduce dependencies to a minimum; to do so, (almost) everything is implemented from scratch (which is the fun part of programming anyway).
Some components (such as online file download), or optimized mathematical routines (matrix multiplication from blas) are offhanded to already existing libraries.
Ideally, such dependencies should be kept to a minimum.

# Contributing
Contributions are welcome! Feel free to open an issue if you find bugs or if you'd like a feature to be implemented, or even better if you'd like
to implement one yourself!
