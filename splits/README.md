# Split files

**The files of this folder are the split of the project.** They are the output of the fourth run of stage
`data` on MIO-TCD (`scripts/make_split.py` at commit `2efbd61`), with no change. Do not make a split again
(design §4.3, "The accepted split").

- **Scene groups:** connected sets of images that are linked at a cosine similarity of 0.7214 or more, in the
  second embedding (the last feature map of the frozen backbone, as a 4 × 4 grid).
- **The guarantee:** no test image and no val image has a train-pool image at or above the limit. The highest
  similarity of a test image to a train-pool image is 0.7212.
- **It is not a camera split.** For 60 of 100 test images that a person looked at, the nearest train-pool image
  shows the same camera in other conditions. This count is a measurement for the report. It is not a pass rule.

**Why this split.** Three methods failed to separate the cameras, in two embeddings (`docs/build_notes.md`
§8, §10 and §12): at one limit, one group holds many cameras, and one camera is in many groups. A random
image split discards the separation that this split has: its test images have near copies in the training
data (a similarity of 1.0000 in the pairs that were looked at). So the answer of the project is for test
images that have no near image in the training data. It is not an answer for new cameras.

| | Images | Scene groups |
|---|---|---|
| test | 10,001 | 1,096 |
| val | 2,017 | 513 |
| train pool | 67,785 | 1,452 |
| train subset | 8,000 | 291 |
| not used | 30,197 | |

| File | Content | SHA-256 starts with |
|---|---|---|
| `test.txt` | MIO-TCD image IDs of the test split: at most 300 of one scene group | `0e6c23878dd86420` |
| `val.txt` | The val split: at most 100 of one scene group | `3b04e94308d1e7b6` |
| `train.txt` | The train subset: 8,000 images of the train pool, the same for all runs | `35570634ff86b82a` |
| `train_pool.txt` | All images of all other scene groups | `7eed9ef168e457f8` |
| `unused.txt` | The images of a test or val group that are over its cap: in no split | `1207b12341cf8e7a` |
| `scene_groups.csv` | The scene group of every labelled image. The bootstrap of the report resamples these groups | `fa9415a232f91b3b` |
| `split_meta.json` | The limit and the record of its search, the caps, groups, images and boxes per class in each split, and the results of the automatic checks | `f33f827175f90a8f` |
| `scene_measurement.json` | The counts of a person on the sheets: the scene pairs (60 of 100), the cameras in 50 images of the largest group (5), and the pairs of a random image split (50 of 50). Not a file of the run | |
| `label_measurement.json` | The counts of a person on the label sheets of 100 random images with 297 boxes: 6 clear objects with no box (2.0%; this count decides the label check of design §4.5), and about 35 at a best guess (about 10.5%). Not a file of the run | |

`tests/test_splits.py` checks the seven hashes and the counts. The notebook reads this folder in stages
`checks` to `report`.

An image ID is the file name without `.jpg`, with its leading zeros. The files hold IDs only. No images are
in this repository.

## Rejected splits

Two folders hold splits that failed the scene check when it was a pass rule (10 pairs of 200 or fewer with the
same camera view). Do not train on them. They are kept as a record.

| Folder | Method | The same camera in the pairs that were looked at |
|---|---|---|
| `rejected_kmeans_k1000/` | Second run (commit `ef89272`): k-means scene groups, k = 1,000, mean embedding | 125 of 125 |
| `rejected_links_mean/` | Third run (commit `016e74f`): groups by links at the limit 0.9913, mean embedding | 71 of 100 |

`rejected_kmeans_k1000/` has no `unused.txt`: the first method gave whole groups to a split.
