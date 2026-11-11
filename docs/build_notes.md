# Build notes

What the build confirmed, corrected and decided. The design is `docs/design.md`. It was changed one time, after
the scene check failed (change 1, 2026-11-10: §4.3, §4.5, §7.3 and §11; §9 below).
Each item gives the section of the design and the place in the code.

Date of the build: 2026-11-10. Library: `transformers` 5.20.1. Checkpoint: `Roboflow/rf-detr-base`.

## 1. What has run, and what has not

| | Where it ran | Result |
|---|---|---|
| Stage 0 checks (`scripts/spike.py`), without the timing | Real checkpoint, CPU | All passed (§2 below) |
| Stage 0 checks with the timing (`scripts/spike.py --time`) | Real checkpoint, Colab T4, fp16 | All passed; speed and memory in §6 |
| Stage 1 (`prepare_data.py`, `make_split.py`), first run | MIO-TCD, Colab T4 | The counts and the ignore classes are confirmed; the val split of the design's size has too few boxes of three classes (§7) |
| Stage 1, second run | MIO-TCD, Colab T4 | The class counts pass with a val split of 5,724 images. **The scene check fails:** the scene groups do not separate the cameras (§8). The split is not accepted |
| The new split method of design change 1 (§9) | Invented cameras (`scripts/scene_split_simulation.py`), CPU | The search, the split and the two automatic checks work, and the failure signals work |
| Stage 1, third run: the new split method with the mean embedding | MIO-TCD, Colab T4 | The search and the automatic checks ran on the GPU and passed. **The scene check fails:** the same camera in 71 of 100 pairs (§10). The split is not accepted |
| Tests (`pytest`) | Tiny random RF-DETR and a synthetic MIO-TCD folder, CPU | 147 passed |
| Tests of the real checkpoint (`pytest -m slow`) | Real checkpoint, CPU | 5 passed |
| The start model on real images | 60 LVIS truck images written in MIO-TCD's CSV format, CPU | AP50 0.30 with no training (§5) |
| Arms H and F on 8 real images | The same stand-in images, CPU | The loss decreases; the checkpoint chain works (§5) |
| The scripts `make_split.py` and `spike.py --image-dir` | The same stand-in images, real backbone, CPU | They run; the sheets look right |
| **Not run:** a training run on a GPU | | Stage 0 timed 23 steps of each arm (§6). Stage `checks` of the notebook is the first run with a data loader, checkpoints and an evaluation |
| **Not run:** the start model and a training run on MIO-TCD | | Stage `checks` of the notebook |

So stage `checks` of the notebook is the first training run on the GPU, and the first run of the
model on MIO-TCD images. It does not start before a split passes the scene check (§8).

## 2. Stage 0: the numbers of design §5.1, on the real checkpoint

| Fact of the design | Real checkpoint | |
|---|---|---|
| Backbone 22,056,576 parameters | 22,056,576 | confirmed |
| Projector 1,444,864 | 1,444,864 | confirmed |
| Decoder 4,687,168 | **4,613,152** | **corrected** |
| Queries and proposals 1,875,952 | 1,875,952 | confirmed |
| Heads (11 classes) 1,896,146 | 1,896,146 | confirmed |
| Total 31,960,706 | **31,886,690** | **corrected** |
| "200 queries is probable" | 300 queries, 13 groups | the default holds |
| Backbone layer 11 has no effect on the output | A change to its `q_proj` changes no logit; layers 0–10 change logits by 2 to 5 | confirmed |
| Inference uses 270,878 head parameters | 270,878 | confirmed |
| Input side a multiple of 56; 560 px | The processor gives 560 × 560 | confirmed |
| `freeze_backbone()` fails | It reads `self.backbone.model`, which does not exist | confirmed; nothing calls it |
| Arm H 1,896,146 trainable | 1,896,146 | confirmed |
| Arms D and B 2,048,210 trainable, 152,064 adapter parameters each | 2,048,210 and 152,064 each (9 and 22 projections) | confirmed |
| Arm F 31,960,706 trainable | **31,886,690** | **corrected** |

The cause of the three corrections is one setting. The checkpoint's decoder samples 2 points for
each attention head (`decoder_n_points = 2`); the library default, which the design measured, is 4.
The layers `sampling_offsets` and `attention_weights` are smaller by 24,672 parameters in each of
the 3 decoder layers. The adapters do not touch these layers, so the budget of §5.3 does not change.

## 3. Differences from the design. Please read these.

**3.1 Ignore regions go under six classes, not seven (§4.4).** The design names seven motorized
classes and says to confirm the list against `localization_evaluation.py`. A copy of that file on
GitHub (`hakimamarouche/MIO-TCD-classification`) has

    motorized_vehicle_classes = ['articulated_truck', 'bus', 'car', 'pickup_truck', 'single_unit_truck', 'work_van']

`motorcycle` is not in it. The code follows the file: `IGNORE_UNDER` in `data/miotcd.py`. A
`motorcycle` detection on a `motorized_vehicle` box is then a false positive. Stage `data` of the
notebook downloads the dataset's code archive and compares the list again
(`prepare_data.py --eval-code`); it stops if they are different. To use the seven of the design,
add `"motorcycle"` to that one tuple, before any training run.

Stage `data` made this comparison with the dataset's own code archive
(`MIO-TCD-Localization-Code.tar`, 2026-11-10): `localization_evaluation.py` lists the same six
classes.

**3.2 The row of `non-motorized_vehicle` gets the prior bias in all 14 class layers (§5.4).** "The
library's default initialisation" gives `class_embed` a bias of −4.6 (a probability of 0.01) and
leaves the 13 proposal copies at a bias of 0. A bias of 0 is a score of 0.5 at every position. The
proposal stage keeps the 300 positions with the highest class score, so this one row would choose
the proposals. The checkpoint's own proposal layers have a bias near −4.6. The code gives the row
weights from N(0, 0.02) and the bias −4.6 in every layer (`arms.replace_class_heads`). The row has
a fixed seed, so the start model is the same for all arms and all seeds.

**3.3 The loss is the library's, called in float32 (§5.1, §6).** The library calculates the loss
inside `forward` when the call has `labels`, in the dtype of the forward pass. Its class loss takes
`log(p)` and `log(1 − p)` of sigmoid outputs. In fp16, `p` is 0 below a logit of about −17 and 1
above about 8, and the loss is NaN or inf. `engine.training_loss` runs the forward pass under
autocast with no `labels` and gives the same tensors to the same loss function
(`model.loss_function`) in float32. A test shows that the two are equal with autocast off. The
truck project did the same.

**3.4 The loss is never called in evaluation mode.** The library's loss always matches 13 query
groups. In evaluation mode the model gives 1 group, and the loss would cut its 300 queries into 13
parts. `training_loss` refuses a model in evaluation mode. There is no val loss; val AP is the val
number.

**3.5 The merge check is in float64 (§5.3, stage 3, stage 6).** In float32, merged backbone
adapters changed a class score by up to 0.44. In float64 the same merge changes it by 3e-15. The
merge is exact; the float32 number shows something else. Merged weights differ from `W x + B A x`
by rounding, the proposal stage keeps the 300 best of some thousand positions, and a rounding
difference in the features changes which positions are kept. The queries of the two models are
then not the same queries. `evaluation.merge_difference` makes the check on a float64 copy on the
CPU. Decoder adapters come after the proposals and do not show the effect. Evaluation uses the
merged model, in float32.

**3.6 No explicit `torch.no_grad()` in arms H and D (§6, "Speed").** With all parameters before
the decoder frozen and an input that needs no gradient, autograd keeps no graph for the backbone
and the projector. The saving is there with no call. A test checks it: the backbone features need
no gradient in H and D, and need one in B and F.

**3.7 "At step 0 each arm gives the output of the start model exactly" (stage 3).** On the real
checkpoint all four arms are bit-equal to the start model. On the tiny test model, arm H differs
from the start model by 1e-6: torch uses another float kernel for a linear layer when its weight is
frozen. So the test asks for what matters: D and B are bit-equal to H (adapters add nothing at step
0), F is bit-equal to the start model, and H is equal to it within 1e-5.

**3.8 Run names and the learning-rate step.** A run folder is `<arm>_lr<rate>_s<seed>`, and
`f_lr<body rate>_h<head rate>_s<seed>` for arm F. "One step in that direction" (§6) is the next
value on 1e-6, 3e-6, 1e-5, …, 1e-2. The choices are in `runs/selection.json`.

## 4. Decisions inside the design

- **Saved predictions hold the 100 best queries of an image** (by top class score), with all 11
  class scores and the box (`pred_<split>.npz`, about 60 MB for the test split). That loses
  nothing: a pair of a query outside the best 100 cannot be among the 100 best pairs, and
  pycocotools reads at most 100 detections per image and class. A test compares the detections
  with `post_process_object_detection(threshold=0, top_k=100)`.
- **pycocotools does the matching; the step from matches to AP is written here**
  (`evaluation.average_precision`). It takes a weight for each image, so a bootstrap resample is a
  weight vector, and 12 runs × 400 resamples need 12 matchings, not 4,800. With weights of 1 it is
  equal to `COCOeval.accumulate` to 1e-12, and with resample weights it is equal to pycocotools on
  a dataset with the images copied. The one difference is the order of detections with equal
  scores, which pycocotools decides by image order.
- **Ignore regions at strict IoU thresholds.** pycocotools matches a detection to a crowd box by
  intersection / detection area, at the same threshold as IoU. At 0.95 a detection that is a little
  larger than the small box is still a false positive. This is the standard behaviour.
- **The representative query** of a ground-truth box is the candidate with the highest IoU (§7.2).
  The truck project took the most confident overlapping query; `errors.py` says why this project
  takes the best box among the 50 candidates.
- **Boxes are `xmax − xmin` wide.** The dataset's own code adds 1 (inclusive pixels). The
  difference is half a pixel and the same for all arms. A box with `xmax = xmin` has zero area here
  and is dropped; the converter counts these.
- **The seed.** Epoch e of seed s takes its image order, flips and dropout masks from (s, e). A
  stopped and resumed run gives bit-equal weights to an uninterrupted run (tested on the CPU).
- **Skipped steps.** `skipped` counts batches with a non-finite loss; stage 6 asks for 0.
  `scaler_skips` counts optimizer steps that the fp16 gradient scaler did not apply. The scaler
  lowers its scale after such a step; a few at the start of a run are normal, and they are in
  `metrics.jsonl`.
- **The scene groups** come from `scipy`'s connected sets of a graph (§9). The first method was scikit-learn's
  k-means, with one start and a fixed seed.
- **Read the pairs of the scene check, not the similarity.** The mean feature of the backbone has a
  large part that all images share: two unrelated photographs had a cosine similarity of about 0.93
  (258 LVIS images). The sheets print the number, but only the two pictures show a camera view.
- **Exit codes.** A script whose check has a negative outcome (a count differs, a split is too small,
  the gate fails) exits with code 3. The notebook goes on, or stops with the design's words, on that
  code only; a crash (1) or a wrong argument (2) stops it as an error.

## 5. The check on real images

MIO-TCD is not on the build machine. The 258 val images of the truck project (LVIS) were written
in MIO-TCD's CSV format (`truck` and `fire_engine` → `single_unit_truck`, `trailer_truck` →
`articulated_truck`) to run the real checkpoint through the real code. These numbers are a check of
the code. They are not results.

- **The start model, with no training,** on 60 of these images: AP 0.25, AP50 0.30
  (`single_unit_truck` 0.46, `pickup_truck` 0.29, `articulated_truck` 0.17), box AP 0.32. The
  resize, the box format and the class rows are correct.
- **Arm H on 8 of these images,** 40 epochs (80 optimizer steps), rate 1e-3: the loss goes from
  23.4 to 4.1 with no skipped batch. The final checkpoint is 7.6 MB (trainable parameters only).
  Loaded again and scored on its 8 training images: AP50 0.74 (the start model: 0.66), AP 0.74
  (0.60). So the chain train → `final.pt` → load → predict → score works with the real weights.
- **Arm F on the same 8 images,** rates 1e-4 for the heads and the body, 16 epochs: the loss goes
  from 21.1 to 10.5. It is slower than arm H because its heads had a tenth of the rate; the
  overfit check now uses 1e-3 for the heads and 3e-5 for the body. The run was stopped after
  epoch 14 and resumed at epoch 15, as the check does it.
- **Not done here:** "arm F reaches AP50 above 0.9 on 16 images" (stage 4). On this CPU one step
  of arm F takes 20 to 45 seconds. Stage `checks` of the notebook runs it on the T4. If it fails
  there and the loss still decreases, give it more epochs (`--epochs-f`).

## 6. Stage 0 on the Colab T4

`scripts/spike.py --time --batch-size 4 --steps 20` at commit `082080b`, on a Tesla T4 with fp16
autocast, 2026-11-10. All checks of stage 0 passed, with the numbers of §2.

| Arm | Images per second | Peak memory (GB) | Hours for one run (96,000 image passes) |
|---|---|---|---|
| H | 28.1 | 0.57 | 0.95 |
| D | 21.7 | 1.29 | 1.23 |
| B | 15.4 | 2.17 | 1.73 |
| F | 13.5 | 3.45 | 1.98 |

- **Batch.** The largest peak is 3.45 GB of the T4's 15 GB. The batch stays 4 images × 4
  accumulation steps (design §6).
- **Time.** Each arm has 5 runs (3 learning rates, then seeds 1 and 2):
  5 × (0.95 + 1.23 + 1.73 + 1.98) = 29.5 hours for the 20 runs. The gate comes after the 3 runs of H
  and the 3 runs of F: 8.8 hours. An added learning-rate step (design §6) is one more run of its
  arm. The design's estimate (§10) was 2 to 3 hours for a run of F or B and half of that for H or D;
  the measured times are at or below it.
- **What the time leaves out.** The table times the training step on one synthetic batch that is
  already on the GPU. A real run also reads and resizes images (2 workers) and evaluates the val
  split at its end. Each epoch line of a run prints its images per second and the time that remains.
- **The loss was finite** in all 23 steps of all arms. The script prints each loss that is not
  finite, and it printed none. So the fp16 forward pass with the float32 loss (§3.3) works on the T4.
- **The gradient scaler left out at least the first optimizer step.** The run gave PyTorch's warning
  "`lr_scheduler.step()` before `optimizer.step()`". The loop steps the optimizer first, so the
  warning has one cause: the scaler did not apply the first step, because a scaled gradient was not
  finite in fp16. The scaler starts with a scale of 65,536 and halves it after each such step, so
  some skipped steps at the start are the normal behaviour. They cost almost nothing: a run has
  6,000 optimizer steps, and the warm-up gives the first three 1% of the learning rate or less. But
  this run did not count them, and a scaler that skips all steps would train nothing. The code now
  counts them:
  - `spike.py --time` has a column "Steps the scaler skipped" and a check that more than half of
    the steps are applied;
  - each epoch line of a run, and the overfit check of stage 4, print the count;
  - the evaluation record of a run has `scaler_skips`, and the report's check table has a column.

  The warning is not shown any more (`engine.step_schedule`); the count replaces it. The check of
  stage 6, "no skipped step", stays as §4 has it: 0 batches with a loss that is not finite.

## 7. Stage 1 on MIO-TCD: the first run

Stage `data` of the notebook at commit `bd06de3`, 2026-11-10. The archive (3.5 GB) came in 64
seconds. The five checks of design §4.5:

**1. Counts: passed.** The CSV has 351,549 rows, 110,000 images, and 20,299 images with a
`motorized_vehicle` row. These are the three reference counts. Each image has a file, no file is
without a row, and no box goes over the edge of its image. 5 rows have zero area; the converter
drops them (design §4.2), and 351,544 boxes remain:

| Class | Boxes | | Class | Boxes |
|---|---|---|---|---|
| `car` | 233,494 | | `pedestrian` | 7,128 |
| `pickup_truck` | 44,282 | | `single_unit_truck` | 5,740 |
| `motorized_vehicle` | 25,845 | | `non-motorized_vehicle` | 2,350 |
| `bus` | 10,598 | | `bicycle` | 2,260 |
| `articulated_truck` | 9,301 | | `motorcycle` | 1,837 |
| `work_van` | 8,709 | | | |

The run printed "counts: DIFFERENT from the reference counts" for the 5 dropped rows. That was an
error of the message, not of the data: the dropped rows are now printed as a note, and the check
compares the counts of the CSV only.

**2. Label completeness: a person counts.** 100 images on 5 sheets. Not recorded here yet.

**3. Class counts: failed at the sizes of the design, for the val split.**

| | Images | Scene groups | Smallest classes |
|---|---|---|---|
| test | 10,088 | 97 | `motorcycle` 195, `non-motorized_vehicle` 234, `bicycle` 448 (100 needed) |
| val | 2,016 | 18 | `bicycle` **5**, `non-motorized_vehicle` **10**, `motorcycle` **29** (30 needed) |
| train pool | 97,896 | 885 | |

A random 2,016 images have about 41 `bicycle` boxes. The val split has 5, because whole scene
groups go to a split and the rare classes are in few scenes. The design's rule is "make the splits
larger". `make_split.py` now does this with the smallest change: a split takes more whole groups,
in the same random order, until each specific class has its minimum (`scenes.split_by_group` with
`class_boxes`). The test split of this run already has its minimums, so the rule changes the val
split and the train pool only. `split_meta.json` records the sizes before and after. The second
run of the stage gives the new size of the val split.

**4. No overlap: passed.** No image and no scene group is in two splits.

**5. Scene check: a person counts.** 200 pairs for the scene split and 200 for a random split. The
mean similarity of a test image and its nearest train-pool image is 0.993 for the scene split and
0.996 for the random split (two random images: 0.914). The count must be made on the sheets of the
second run, because the train pool changes when the val split grows.

Other results of the run:

- **The image IDs do not show the camera.** Images next to each other in ID order have a mean
  similarity of 0.914, and random pairs have 0.914. So the scene groups are clusters (design §4.3).
- **The ignore classes are the six of §3.1**, confirmed against the dataset's code archive.
- **The train subset** (8,000 images) of this first split had 139 `bicycle`, 149 `motorcycle` and
  171 `non-motorized_vehicle` boxes. The second run draws a new subset from the smaller train pool.

## 8. Stage 1 on MIO-TCD: the second run

Stage `data` of the notebook at commit `ef89272`, 2026-11-10. The counts and the ignore classes are
as in the first run.

**Class counts: passed, with a larger val split.**

| | Images | Scene groups | Smallest classes |
|---|---|---|---|
| test | 10,088 | 97 | `motorcycle` 195, `non-motorized_vehicle` 234, `bicycle` 448 (100 needed) |
| val | 5,724 | 49 | `bicycle` 30, `non-motorized_vehicle` 69, `motorcycle` 90 (30 needed) |
| train pool | 94,188 | 854 | |
| train subset | 8,000 | | `motorcycle` 131, `bicycle` 137, `non-motorized_vehicle` 185 |

The val split took 31 more groups to get 30 `bicycle` boxes. It is 2.8 times the size of the
design, so the scoring of the val split after each run takes 2.8 times as long.

**No overlap: passed.**

**Label completeness: counted; the result depends on what counts as an object.** The 100 images
have 297 boxes. The count of the objects with no box, by a person:

| What is counted | With no box | Rate |
|---|---|---|
| Clear objects: a vehicle that is fully visible, moving or on the road | 6 | 6 of 303 = 2.0% |
| All objects at a best guess: also tiny, occluded and blurry vehicles, and headlights at night | about 35 | 35 of 332 = about 10.5% |

No pedestrian with no box was found. The limit of design §4.5 is 5%: the first count is below it,
and the second is above it. The design does not say which objects count. This decision is open.

**Scene check: failed.** A person looked at 5 of the sheets of the scene split, 125 pairs. Each
pair shows the same camera view. The limit is 10 pairs of 200. So the k-means groups do not
separate the cameras: a test image has a frame of its own camera in the train pool. The numbers of
both runs showed this before the count: the mean similarity of a test image and its nearest
train-pool image is 0.993 for the scene split and 0.996 for a random split, so the scene split
removed almost nothing. The notes of the first run recorded the two numbers and did not draw this
conclusion.

A likely cause, not measured yet: k-means cuts the frames of one camera into more than one
cluster (by the light, the weather or the traffic), so a frame in a test cluster has a near copy
in a cluster of the train pool.

The split of this run is not accepted, and no training starts on it. Its files are in
`splits/rejected_kmeans_k1000/` as a record; `splits/` itself stays empty until a split passes.
The design gives no rule for a failed scene check, so the next method is a change to design §4.3.

**COCO detections: looked at.** On the sheet of 20 images, each box of the checkpoint is on its
object, and no class name is wrong.

## 9. Design change 1: scene groups by links

The scene check failed with the k-means groups (§8). The design is changed in four places: §4.3
(scene groups, the search for the limit, the split with caps, what to do if no limit passes), §4.5
(two automatic checks before the scene check), §7.3 (the bootstrap resamples scene groups) and §11
(limits). This section gives what the build did with the change.

**What is built.**

| Design | Code |
|---|---|
| Scene groups for a limit (§4.3) | `scenes.neighbour_lists`, `scenes.link_groups` |
| Candidate limits and the search | `scenes.link_distances`, `scenes.candidate_limits`, `scenes.search_limit` |
| The split with a cap for each group, and `unused.txt` | `scenes.split_with_caps`, `scenes.split_problems` |
| The group check and the split check (§4.5) | `scenes.group_check`, `scenes.split_check` |
| The second embedding | `scenes.embed_images(grid=4)`, `make_split.py --embedding grid`, `EMBEDDING` in the notebook |
| The bootstrap of scene groups (§7.3) | `stats.group_resample_weights`; the gate (`runs.run_gate`) and the report (`report.analyse`) read `scene_groups.csv` |
| The image interval of AP_B − AP_D, one time | "The unit of the bootstrap" in the report, under "Differences in AP" |
| The limits of §11 | The "Limits" section of the report |
| The test on invented cameras | `scripts/scene_split_simulation.py`; a small copy runs in the tests |

`cluster_scenes` and `split_by_group` are gone. scikit-learn is no longer a dependency of the
package; the k-means line of the simulation needs it (`pip install -e ".[dev]"`).

**Decisions inside the change.**

- **One similarity for a pair.** Each pass over all pairs calculates the same blocks of rows
  (`unit[start:stop] @ unit.T`, 2,048 rows), in float32. So a pair has the same similarity in the
  neighbour lists, in step 3 and in the two checks. The limit is a float32 number too. The groups
  are a function of the embeddings and the limit; the rounding of a similarity that is equal to
  the limit is the one part that can depend on the machine.
- **Step 3 looks at all images,** not only at those with a full neighbour list. Its last pass
  finds nothing, and that pass is the group check. `make_split.py` runs the group check one more
  time on the groups that it writes.
- **Group numbers** are the rank of the first image of the group, so they do not depend on the code
  that found the groups.
- **d_rand** is the median of N pairs (N images) with a random shift of the image position, so no
  image is a pair with itself. The seed is the split seed.
- **The draw of a group** (at most 300 or 100 images) has its own generator, from the split seed and
  the place of the group in the sorted group numbers. The draw of a group does not depend on the
  groups before it.
- **The minimum of groups** of the split check is ⌈images / cap⌉: 34 and 20 at the sizes of the
  design.
- **No limit: no split.** `make_split.py` then removes the split lists of an earlier run from the
  splits folder (the six file names that it writes, and no other file), writes the record of the
  search to `split_meta.json`, and exits with code 3. The notebook does not start a later stage on
  a splits folder whose `split_meta.json` has no limit or a failed automatic check. So the rejected
  k-means split on Drive cannot be used by mistake.
- **A sheet of the largest group.** `make_split.py` writes 50 images of the largest group to
  `largest_group.jpg`, for the limit "If the largest group is near 20%, look at 50 of its images".
- **Groups from a file** (`--groups-csv`, design §4.3 "if the image IDs show the camera") get the
  split with caps. The group check and the split check need a limit, so they do not run.
- **The bootstrap needs `scene_groups.csv`** in the splits folder. With no file the gate and the
  report stop with a message. With one image in each group the weights are those of the image
  bootstrap.
- **The image interval** is calculated for the runs of arms B and D only (6 of the 12 runs).

**What was run.**

- The four cases of the simulation, with the code of the project. The results are the table of
  design §4.3: the chosen limits are 0.9689, 0.9837 and 0.9954, and case D has no limit. At each
  limit the groups are equal to those of the first version of the script, which had its own code.
- 147 tests: the groups against a brute-force result with neighbour lists that are too short; the
  search with its two causes of "over-linked"; the caps; the two checks on a split with a planted
  failure; the group weights; `make_split.py` on planted cameras, with no limit, with the grid
  embedding and with groups from a file; the gate and the report with the group bootstrap.

- The two embeddings of 12 photographs with the real backbone, on the CPU. At 224 px the last
  feature map has 16 × 16 positions and 384 channels, so the grid embedding has 6,144 numbers, as
  the design says. Its largest value is 21.6, so float16 holds it. Two unrelated photographs have a
  similarity of 0.86 with the mean embedding and −0.09 with the grid embedding (centred).
- The time of the search on 29,269 invented frames of 384 numbers, on the CPU of the Mac: 1.1
  seconds for one pass over all pairs, and one pass for each limit. At 110,000 images a pass has 14
  times as many pairs. On the T4 the whole search is expected to take a few minutes; this is not
  measured.

**What was not run.** MIO-TCD. Only the scene check on the real images shows if the cameras are
separable in the real embedding. The search and the checks ran on a CPU only, not on a GPU. The
grid embedding was not used for a split of real images.

**A result of the simulation to keep in mind.** With 40 invented cameras that are not separable,
the search accepts its first candidate, both automatic checks pass, and the same camera is in 85%
of the scene-check pairs. The automatic checks show that the code did what the limit asks. They do
not show that the limit is low enough. The scene check by a person decides.

## 10. Stage 1 on MIO-TCD: the third run (groups by links, mean embedding)

Stage `data` of the notebook at commit `016e74f`, 2026-11-10, on the T4. This was the first run of the
search and the two automatic checks on a GPU, and the first on real images. They ran with no error.

**The search.** The median similarity to the nearest image is 0.9967, and that of random pairs is 0.9244.

| Limit | Groups | The largest group | |
|---|---|---|---|
| 0.9967 | 65,465 | 5.0% of the images | |
| 0.9960 | 51,906 | 6.8% | |
| 0.9952 | 38,965 | 7.3% | |
| 0.9941 | 27,363 | 7.6% | |
| 0.9928 | 18,605 | 9.5% | |
| **0.9913** | 12,098 | 19.7% | chosen: the lowest that is not over-linked |
| 0.9894 | 7,142 | 58.0% | over-linked |

**The split at the limit 0.9913.**

| | Images | Scene groups |
|---|---|---|
| test | 10,000 | 3,648 |
| val | 3,683 | 1,922 |
| train pool | 45,792 | 6,528 |
| train subset | 8,000 | |
| not used (over the cap of a test or val group) | 50,525 | 23 |

The smallest classes: `motorcycle` 162 and `non-motorized_vehicle` 196 in test (100 needed);
`motorcycle` 30 and `non-motorized_vehicle` 93 in val (30 needed).

**The automatic checks: passed.** No overlap, class counts, the group check (0 failures: the highest
similarity of an image to an image outside its group is 0.9912682, just below the limit of
0.9912685) and the split check (0 failures: the highest similarity of a test image to a train-pool
image is 0.9912674).

**The scene check: failed.** A person looked at 4 of the 8 sheets of the scene split, 100 pairs:

| | Pairs |
|---|---|
| The same camera view | 71 |
| Different camera views | 27 |
| Not clear (dark; likely the same) | 2 |

The limit is 10 pairs of 200. The pairs have a mean similarity of 0.984, and all are below the limit
of 0.9913, as the split check says. On the sheets of the random image split, 50 of the 50 pairs that
were looked at show the same camera view.

**The largest group** has 21,715 images (19.7%), so the design asks for a look at 50 of them. The 50
images show 14 cameras. One camera has 16 of the 50, two have 6 each, two have 4 each, and five have
1 each.

**What the two looks show together.** At one limit, both things are true:

- images of many cameras are in one group, so there are pairs of two cameras at or above 0.9913;
- 71% of the test images have a frame of their own camera in the train pool, so there are pairs of
  one camera below 0.9913.

So the mean embedding has no limit that separates the cameras: a higher limit cuts more cameras, and
a lower one joins more. This is the second failure signal of design §4.3: the search accepts a
limit, the automatic checks pass, and the scene check fails. The design's next step is the second
embedding (the 4 × 4 grid), one time.

**The sizes of the groups** at this limit, from `scene_groups.csv`:

| Images in the group | Groups | Share of all images |
|---|---|---|
| 1 | 9,442 | 8.6% |
| 2 to 9 | 2,550 | 5.9% |
| 10 to 99 | 47 | 1.2% |
| 100 or more | 59 | 84.3% |

The 12 largest groups have 59% of the images (21,715; 11,095; 5,398; 4,514; 4,480; ...). The caps and
these sizes change the content of the test split more than the design's limit note suggests: 15
groups over the cap give 4,500 of its 10,000 images, and groups of one image give 2,835. A group of
one image is 8.6% of the dataset and 28% of the test split. This is a number for the next split too:
if its groups have such sizes, the test split has many frames that are not near any other frame.

The split files of this run are in `splits/rejected_links_mean/`, as a record.
