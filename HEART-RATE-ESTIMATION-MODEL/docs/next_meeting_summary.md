# Next Meeting Summary

I added a modular PyTorch pipeline for wearable PPG heart-rate estimation:

Raw PPG -> preprocessing -> frozen TinyPPG artifact detection -> crop noisy regions -> variable-length clean PPG -> trainable HR model -> bpm.

TinyPPG is not trained. It is loaded from the existing `Tiny-PPG-master` code/checkpoint, frozen, and used only to produce an artifact mask. The HR estimator is the only trainable part.

The main open item is confirming TinyPPG output semantics for any checkpoint other than the local `output["seg"]` checkpoint. Real PPG-DaLiA training also still depends on having the local dataset prepared into CSV windows.

