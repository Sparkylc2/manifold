
build toggle (change is sticky)
`cmake -B build -DMANIFOLD_DEMOS="all"` (or the filename of the header without the .h)

compile and run
`cmake --build build -j8 ; ./build/demos/launcher/manifold`


## AI checkpoints

The forecast cell (`karman_forecast`, and the bottom slot of story frame 4) plays
straight from `assets/ai/forecast_demo_weights.bin` — it never trains at runtime.
Only rerun the trainer if the flow, the grid, or a number in
`demos/forecast_demo_spec.h` changes. A checkpoint that disagrees with the spec
is rejected by the shape header at the front of the file, and the cell says so
rather than drawing garbage.

`cmake --build build -j8 --target train_forecast_demo`

`./build/train_forecast_demo [n_frames] [lstm_epochs] [out] [warmup_s] [pod_frames] [ae_epochs] [ae_out]`

Defaults are `800 400 assets/ai/forecast_demo_weights.bin 20 320 1400
assets/ai/ae_demo_weights.bin`. The POD/ESN/LSTM half takes about two minutes,
most of it the fluid; the autoencoder adds ~20 more and is written to its own
file, so the AE cell does not have to read the forecast models to reach its own
weights. Both cells reject a checkpoint that disagrees with the spec. `warmup_s` is sim seconds spent letting the street
develop BEFORE the first snapshot is recorded — fitting a basis to the transient
biases everything downstream. It prints a drift check on the recorded window;
anything above 8% means raise it. `pod_frames` caps how many snapshots the SVD
itself sees (strided across the whole window, so it still sees every phase of the
cycle) because a thin SVD costs O(cols²) in time and memory; the coefficient
series the forecasters fit against always uses all `n_frames`.

It reports the per-mode energy split, the POD and AE reconstruction errors over
the full window, and a closed-loop rollout score per model, then reloads both
files it wrote and verifies the weights round-trip bit for bit.

Anything driving these cells must advance them at a fixed **1/60 s** step
(`--dt 1/60`). Both forecasters were fitted at one step per 0.05 s of simulated
time; the cells now accumulate sim time rather than counting frames, so a
different rate no longer races them ahead of the flow, but 1/60 is what the
recording path assumes.
