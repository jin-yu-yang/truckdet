# Split files

Stage `data` of the notebook writes these files (`scripts/make_split.py`); commit them here.

| File | Content |
|---|---|
| `test.txt` | MIO-TCD image IDs of the test split: whole scene groups, 10,000 images or more |
| `val.txt` | The val split: whole scene groups, 2,000 images or more |
| `train_pool.txt` | All other labelled images |
| `train.txt` | The train subset: 8,000 images of the train pool, the same for all runs |
| `scene_groups.csv` | The scene group of every labelled image |
| `split_meta.json` | Seed, group count, images and boxes per class in each split |

An image ID is the file name without `.jpg`, with its leading zeros. The files hold IDs only. No images are
in this repository.
