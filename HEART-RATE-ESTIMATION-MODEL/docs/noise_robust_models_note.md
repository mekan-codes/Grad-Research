# Noise-Robust PPG HR Models

Wearable PPG noise usually means corrupted optical pulse measurements. It often comes from motion, loose sensor contact, skin pressure changes, ambient light leakage, sweat, and low perfusion.

PPG is noisy because the signal is small and the wrist is a difficult measurement site. Arm motion changes the optical path, and motion artifacts can land in the same frequency band as heart beats.

A noise-robust model should avoid trusting every sample equally. Useful approaches include:

- 1D CNNs for local pulse morphology.
- CNN + GRU/LSTM models for beat-to-beat temporal context.
- Temporal CNNs for longer receptive fields with lightweight inference.
- Transformers with masks so padded or removed samples are ignored.
- Uncertainty-aware HR estimation to express low confidence on corrupted windows.

The chosen pipeline is robust because TinyPPG removes motion artifact first, cropping prevents corrupted regions from influencing HR estimation, and the HR model uses masked variable-length pooling. The stronger model adds temporal context with a GRU and masked attention. Accelerometer features can be added later to help identify motion-driven artifacts.

