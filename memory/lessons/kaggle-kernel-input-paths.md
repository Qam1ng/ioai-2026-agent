# Kaggle kernel data paths are nested — never hardcode

Competition data mounts at /kaggle/input/competitions/<slug>/ (one level deeper
than /kaggle/input/<slug>/). Always DISCOVER paths at runtime: os.walk for the
dir containing the sample submission, and for checkpoints look for config.json +
*.safetensors. Hardcoded HF-style paths make from_pretrained treat the path as a
hub repo id and fail (internet is off in scoring kernels).
