# Security

## What this project does with your data

- The web demo is a static page. The camera feed is processed **only in your browser** (WebGPU / WebAssembly). No image, video or measurement is sent anywhere, and there is no backend, analytics or cookie.
- The model files are fetched from this repository's GitHub Release and from Google's MediaPipe storage. `npm run models` checks their SHA-256 sums.

## Loading weights

PyTorch checkpoints are loaded with `torch.load(..., weights_only=True)`, so a malicious `.pt` file cannot run code. Exception: `student/train.py` resumes from its own `last.pt` with `weights_only=False`, so only resume from runs you created yourself.

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** (Security tab → "Report a vulnerability") rather than a public issue.
