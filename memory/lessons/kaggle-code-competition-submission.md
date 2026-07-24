# Code competitions only accept notebook submissions

Raw CSV upload returns 400 "This competition only accepts Submissions from
Notebooks." Path that works: build a kernel dir with kernel-metadata.json
(competition_sources set), `kaggle kernels push`, wait for COMPLETE, then
submit the kernel's output via competition_submit_code (file_name=submission.csv,
kernel ref + version). Score appears asynchronously (can stay PENDING ~30+ min).
