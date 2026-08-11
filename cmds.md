
build toggle (change is sticky)
`cmake -B build -DMANIFOLD_DEMOS="all"` (or the filename of the header without the .h)

compile and run
`cmake --build build -j8 ; ./build/demos/launcher/manifold`
