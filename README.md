# Morphophonetics

Code for AMP2025 submission for exemplar-based production from mel spectrograms.
Running this code requires access to the METU Turkish speech corpus from LDC (LDC2006S33).

The code in `utils.py` prepares the data, given the above corpus, segmenting words using
the provided alignments, and computing mel spectrograms. The experiment runner
demonstrates simple production from segmented tokens, as well as generalization of
vowel harmony to held-out meanings.

