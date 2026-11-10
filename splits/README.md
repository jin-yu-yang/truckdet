# Split files

Stage `data` of the notebook writes these files (`scripts/make_split.py`); commit them here.

**There is no accepted split yet.** `rejected_kmeans_k1000/` holds the split of the second run on MIO-TCD
(2026-11-10, commit `ef89272`): k-means scene groups, k = 1,000, test 10,088 images, val 5,724, train pool 94,188,
train subset 8,000. It failed the scene check of design §4.5: the nearest train-pool image of a test image shows
the same camera view in all 125 pairs that were looked at (the limit is 10 of 200). Do not train on it. It is kept
as a record; see `docs/build_notes.md` §8.

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
