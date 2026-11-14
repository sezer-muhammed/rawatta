# Security

Please report security problems privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue for them.

Supported: the latest release and `main`.

Relevant areas: the weight converter (`tools/convert_weights.py` reads PyTorch checkpoints with a restricted unpickler
and must never execute code from a file), parsing of safetensors headers and of input images and videos by
`rawatta`, the download script (it must never print or expose the Hugging Face token), and the Docker image.
Problems in VGGT-Ω itself or in its weights belong to [Meta's repository](https://github.com/facebookresearch/vggt-omega).
