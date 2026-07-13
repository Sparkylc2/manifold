# AI layer tests

Standalone gradient-check + end-to-end tests for the conv/dense layers and the CAE.
Build & run (no CMake needed):

    g++ -std=c++20 -O2 -I include -I build/_deps/eigen-src \
      tests/test_ai.cpp \
      src/ai/conv_layer.cpp src/ai/dense_layer.cpp src/ai/layer.cpp src/ai/conv_autoencoder.cpp \
      -o /tmp/test_ai && /tmp/test_ai

Checks:
- ConvolutionalLayer (normal, transposed, ReLU): gK / gb / dX via finite differences
- DenseLayer: gW / gb / dX
- ConvolutionalAutoencoder: encode/decode shape round-trip + overfit a tiny set
