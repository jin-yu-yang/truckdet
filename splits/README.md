# Split files

Stage `data` of the notebook writes these files (`scripts/make_split.py`); commit them here.

**There is no accepted split yet.** `rejected_kmeans_k1000/` holds the split of the second run on MIO-TCD
(2026-11-10, commit `ef89272`): k-means scene groups, k = 1,000, test 10,088 images, val 5,724, train pool 94,188,
train subset 8,000. It failed the scene check of design §4.5: the nearest train-pool image of a test image shows
the same camera view in all 125 pairs that were looked at (the limit is 10 of 200). Do not train on it. It is kept
as a record; see `docs/build_notes.md` §8.

| File | Content |
|---|---|
| `test.txt` | MIO-TCD image IDs of the test split: 10,000 images or more, at most 300 of one scene group |
| `val.txt` | The val split: 2,000 images or more, at most 100 of one scene group |
| `train_pool.txt` | All images of all other scene groups |
| `train.txt` | The train subset: 8,000 images of the train pool, the same for all runs |
| `unused.txt` | The images of a test or val group that are over its cap: in no split |
| `scene_groups.csv` | The scene group of every labelled image. The bootstrap of the report resamples these groups |
| `split_meta.json` | The limit and the record of its search, the caps, groups, images and boxes per class in each split, and the results of the automatic checks |

A scene group is a connected set of images that are linked at a cosine similarity at or above a limit (design
§4.3). `rejected_kmeans_k1000/` has no `unused.txt`: the first method gave whole groups to a split.

An image ID is the file name without `.jpg`, with its leading zeros. The files hold IDs only. No images are
in this repository.
