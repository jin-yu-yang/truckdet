# Split files

Stage `data` of the notebook writes these files (`scripts/make_split.py`); commit them here.

**There is no accepted split yet.** Three folders hold splits that failed the scene check of design §4.5. Do not
train on them. They are kept as a record; see `docs/build_notes.md` §8, §10 and §12. Both embeddings of
design §4.3 have failed, so the design asks for a decision: a random image split, or stop.

| Folder | Method | Result of the scene check |
|---|---|---|
| `rejected_kmeans_k1000/` | Second run on MIO-TCD (commit `ef89272`): k-means scene groups, k = 1,000 | The same camera in 125 of the 125 pairs that were looked at (the limit is 10 of 200) |
| `rejected_links_mean/` | Third run (commit `016e74f`): groups by links at the limit 0.9913, mean embedding | The same camera in 71 of the 100 pairs that were looked at |
| `rejected_links_grid/` | Fourth run (commit `2efbd61`): groups by links at the limit 0.7214, the second embedding (4 × 4 grid) | The same camera in 60 of the 100 pairs that were looked at |

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
