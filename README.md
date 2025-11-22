# Tiny Tensor Library (TTL)

TTL is designed to be a lightweight and easy to use library for tensors. 
This is achieved by supporting only one datatype, F32.
The library tries to support _most_ of what numpy APIs, with the goal to use this library as the backend for a full python library in its v2. 

# Dependencies
TTL tries to be free of dependencies; to do so, everything is implemented from scratch (which is the fun part of programming anyway).
As an example, its often the case that a user might want to fetch a dataset online (see MNIST for an example). Instead of relying 
on external dependencies (such as libcurl, which is OS dependent), TTL implements a small HTTPS library to do so (see utils/https.h).
The same philosophy is applied to other tools of the library.

# Contributing
Contributions are welcome! Feel free to open an issue if you find bugs or if you'd like a feature to be implemented, or even better if you'd like
to implement one yourself!

# Note
Currently code still relies on external dependencies for foundation testing purposes. In the future, said dependencies will be removed.