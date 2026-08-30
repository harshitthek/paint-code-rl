# Datasets

This directory is the expected location for:
1. Static reference image pools
2. Validation datasets (prompt corpora)

Large generated datasets or raw model weights should **not** be committed directly to git.

### Expected Formats
* **Reference Pool:** Flat directory of .png files representing target aesthetic states.
* **Prompt Corpora:** .jsonl files where each line contains a {"prompt": "..."} mapping.

**Do not commit private data without legal review.**
