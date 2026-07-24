# Submit to a Kaggle CODE competition (notebook-scored)

1) Build kernel dir: <name>.py + kernel-metadata.json:
   {"id":"<user>/<slug-kernel>","title":"...","code_file":"<name>.py",
    "language":"python","kernel_type":"script","is_private":true,
    "enable_gpu":true,"enable_internet":false,
    "competition_sources":["<competition-slug>"],"dataset_sources":[],"kernel_sources":[]}
2) In the script: DISCOVER paths (os.walk /kaggle/input for sample submission;
   checkpoint = dir with config.json + *.safetensors). Write output to
   /kaggle/working/submission.csv with exactly the sample's columns/row order.
3) kaggle_push_kernel(kernel_dir) -> note version number in output.
4) Poll kaggle_kernel_status until COMPLETE (run_bash 'sleep 30' between polls;
   ERROR -> kaggle_kernel_log, fix, re-push).
5) kaggle_submit(file_name="submission.csv", kernel_version=<ver>).
6) kaggle_submissions to read the score (may be PENDING for a long while —
   proceed with other work and re-check).
Gotchas: scoring kernel has NO internet; GPU may not always be granted (write
device-agnostic code); each push reruns ~feature extraction, keep kernels lean.
