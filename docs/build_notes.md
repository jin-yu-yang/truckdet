# Build notes

What the build confirmed, corrected and decided. The design is `docs/design.md`. It was changed two times. Change 1
(2026-11-10: §4.3, §4.5, §7.3 and §11; §9 below) came after the scene check failed with k-means. Change 2
(2026-11-10: §3, §4.3, §4.5, §11 and §12; §13 below) accepts the split of the fourth run.
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
| Stage 1, fourth run: the new split method with the second embedding (the 4 × 4 grid) | MIO-TCD, Colab T4 | The search and the automatic checks passed. **The scene check fails:** the same camera in 60 of 100 pairs (§12). The split was not accepted then. Design change 2 accepts it with its scene measurement (§13) |
| Design change 2: the split of the fourth run is the split of the project (§13) | The split files of the repository, CPU | The seven files have the hashes of the change, and its numbers agree with the files. The report gives the scope of the answer and the new limits |
| Label completeness (stage 1), with the decision of §14 | A person, 100 random images of MIO-TCD | **Passed:** 2.0% of the clear objects have no box (the limit is 5%). About 10.5% with the objects that a person can only guess; this count goes in the report |
| Tests (`pytest`) | Tiny random RF-DETR and a synthetic MIO-TCD folder, CPU | 155 passed |
| Tests of the real checkpoint (`pytest -m slow`) | Real checkpoint, CPU | 5 passed |
| The start model on real images | 60 LVIS truck images written in MIO-TCD's CSV format, CPU | AP50 0.30 with no training (§5) |
| Arms H and F on 8 real images | The same stand-in images, CPU | The loss decreases; the checkpoint chain works (§5) |
| The scripts `make_split.py` and `spike.py --image-dir` | The same stand-in images, real backbone, CPU | They run; the sheets look right |
| Stages 2 to 4 (`evaluate.py --start-model`, `overfit_check.py`): stage `checks` of the notebook | MIO-TCD, Colab T4, fp16 | **All passed** (§15). The start model on the val split: AP50 0.77 for `car` and 0.85 for `bus`. On 16 images, arm F reaches AP50 1.000, the loss of H, D and B decreases, and a stopped run resumes |
| Stage 5: the learning-rate runs of H and F, and the gate: stage `gate` of the notebook | MIO-TCD, Colab T4, fp16 | **The gate passes** (§16): AP_F − AP_H = +0.172 on the val split, 12.0 times its standard error. 8 runs of 12 epochs, each with 0 skipped batches. H chose the rate 1e-3; F chose 1e-5 for the body |
| **Not run:** a run of D or of B, seeds 1 and 2, the test split and the report | | Stages `main` and `report` of the notebook. The scripts ran in the tests, on the tiny model |

The split is accepted (§13), the label check passes with the count of clear objects (§14), and
stages 2 to 4 passed on MIO-TCD (§15). The gate passed (§16). The next stage is `main`.

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
and the second is above it. The design did not say which objects count. Decided on 2026-11-10 (§14):
the clear objects decide the check, so it passes with 2.0%.

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

## 11. The notebook took the code archive for the dataset

The fourth run of stage `data` (for the grid embedding) stopped before the split, with
"No such file or directory: /content/data/code/MIO-TCD-Localization-Code/train/00000301.jpg".

- **Cause.** The dataset cell of the notebook took the first `gt_train.csv` under `/content/data`
  as the dataset. Stage `data` extracts the dataset's code archive to `/content/data/code`, and
  that archive has a `gt_train.csv` too. In a new session the archive is not there yet when the
  cell runs, so runs 1 to 3 were correct. This run used a session that had the archive from the
  run before it, and the cell took the folder of the archive.
- **What the run did before it stopped.** It copied the cached `annotations.json` and `counts.json`
  into the folder of the archive and printed the counts from them. Nothing was written to Drive: the
  split files and the embeddings are as they were.
- **Fix.** A folder is the dataset only if it has `gt_train.csv` and, in `train/`, an image for
  each image ID of the CSV (`is_dataset` in the dataset cell). This also makes the cell extract the
  archive again after an extraction that was stopped in the middle. The fix is in the notebook
  file, so open the notebook from GitHub again; the cell that pulls the code does not update the
  notebook.

## 12. Stage 1 on MIO-TCD: the fourth run, done again (groups by links, grid embedding)

The fourth run stopped before the split (§11). This is the same run, done again with the notebook at
commit `2efbd61`, 2026-11-10, on the T4, with `EMBEDDING = "grid"`. It is the second embedding of
design §4.3, and the design allows it one time. The dataset cell took the correct folder
(`/content/data/MIO-TCD-Localization`).

**The embedding.** 6,144 numbers for each image (the mean of the last feature map in each cell of a
4 × 4 grid), with the mean of all images subtracted. The median similarity to the nearest image is
0.8980, and that of random pairs is −0.0137. For the mean embedding these were 0.9967 and 0.9244, so
the grid embedding uses a much larger part of the scale. Images next to each other in ID order have a
mean similarity of 0.003, and random pairs 0.001: the IDs do not show the camera.

**The search.**

| Limit | Groups | The largest group | |
|---|---|---|---|
| 0.8980 | 67,367 | 1.5% of the images | |
| 0.8822 | 55,208 | 2.0% | |
| 0.8641 | 42,582 | 5.2% | |
| 0.8431 | 30,707 | 6.5% | |
| 0.8189 | 20,322 | 7.3% | |
| 0.7909 | 12,008 | 7.5% | |
| 0.7587 | 6,322 | 7.7% | |
| **0.7214** | 3,061 | 11.5% | chosen: the lowest that is not over-linked |
| 0.6784 | 1,322 | 21.5% | over-linked |

**The split at the limit 0.7214.**

| | Images | Scene groups |
|---|---|---|
| test | 10,001 | 1,096 |
| val | 2,017 | 513 |
| train pool | 67,785 | 1,452 |
| train subset | 8,000 | |
| not used (over the cap of a test or val group) | 30,197 | 29 |

The smallest classes: `motorcycle` 148 and `non-motorized_vehicle` 257 in test (100 needed);
`motorcycle` 45 and `non-motorized_vehicle` 48 in val (30 needed).

**The automatic checks: passed.** No overlap, class counts, the group check (0 failures: the highest
similarity of an image to an image outside its group is 0.721404, just below the limit of 0.721408)
and the split check (0 failures: the highest similarity of a test image to a train-pool image is
0.7212, of a val image to a train-pool image 0.7212, and of a test image to a val image 0.7210).

**The scene check: failed.** A person looked at 4 of the 8 sheets of the scene split, 100 pairs:

| Sheet | The same camera view | Different camera views |
|---|---|---|
| `scene_split_01.jpg` | 14 | 11 |
| `scene_split_02.jpg` | 13 | 12 |
| `scene_split_03.jpg` | 17 | 8 |
| `scene_split_04.jpg` | 16 | 9 |
| **All** | **60** | **40** |

The limit is 10 pairs of 200. The pairs have a mean similarity of 0.554, and the highest is 0.7196,
below the limit of 0.7214, as the split check says. The pairs of the random image split have a mean
similarity of 0.885, and the highest is 1.0000: the dataset has pairs of images that are equal to 4
decimal places in this embedding. The sheets of the random image split were not counted in this run.

The 60 is a lower limit for the share of test images whose camera is in the train pool. A pair
shows only the nearest train-pool image. If that image shows a different camera, a frame of the
test image's camera can still be in the train pool, at a lower similarity.

**The largest group** has 12,603 images (11.5%), and it is in the train pool. Its share is not near
20%, but a person looked at the 50 images of `largest_group.jpg`. They show 5 cameras: one camera
has 26 of the 50, and the others have 10, 9, 4 and 1. The camera with 4 is a dark road with no
street light, which the person had seen on the sheets of an earlier run. Some of its images are
very dark; they are counted as this camera because the centre lines and the angle are the same.
The largest group of the third run showed 14 cameras in 50 images.

**What the two looks show together.** As in §10, both things are true at one limit:

- images of 5 cameras or more are in one group, so there are pairs of two cameras at or above 0.7214;
- 60% of the test images or more have a frame of their own camera in the train pool, so there are
  pairs of one camera below 0.7214.

The grid embedding is better than the mean embedding on both counts, but it is far from the limit
of the scene check (5%):

| Split | Scene check: pairs that show the same camera | The largest group: cameras in 50 images |
|---|---|---|
| K-means, k = 1,000, mean embedding (§8) | 125 of 125 | not looked at |
| Links, mean embedding, limit 0.9913 (§10) | 71 of 100, and 2 not clear | 14 |
| Links, grid embedding, limit 0.7214 (this run) | 60 of 100 | 5 |
| Random image split (the sheets of the third run) | 50 of 50 | |

**The result for the design.** Both embeddings failed the scene check. This is step 2 of "If no
limit passes" in design §4.3: the cameras are not separable with these tools, and a decision is
needed. The alternatives of the design are: continue with a random image split (the design
recommends this), or stop. No other limit and no third embedding was tried, as the design says.

**The sizes of the groups** at this limit, from `scene_groups.csv`:

| Images in the group | Groups | Share of all images |
|---|---|---|
| 1 | 2,082 | 1.9% |
| 2 to 9 | 818 | 2.1% |
| 10 to 99 | 83 | 2.4% |
| 100 or more | 78 | 93.6% |

The 12 largest groups have 49% of the images (12,603; 8,477; 5,205; 4,790; 3,563; ...). In the test
split, 22 groups over the cap give 6,600 of its 10,001 images, and groups of one image give 757.

**Open, with the decision: the unit of the bootstrap** (closed by design change 2, §13: the split of
this run is accepted, so the unit is its scene group). Design §7.3 resamples the scene groups of the
split. A random image split has no scene groups of its own, and `runs.split_groups` stops if
`scene_groups.csv` is not there. The design does not say what the unit is in that case. Nothing is
built for this yet. The numbers for the decision:

- The groups of this run can be the unit. They are not cameras, but each group holds frames that
  are near each other, and a group that joins cameras makes an interval wider, not narrower.
- With these groups, a random test split of 10,000 images (seed 0) has images of 522 groups. The
  largest group gives 1,139 of them (11.4%), and the next give 723, 471, 432 and 348.
- A rough measure of the number of independent units is the square of the image count, divided by
  the sum of the squares of the group counts. It is 30 for that random test split, and 46 for the
  test split of this run (the caps make the groups more equal). So the intervals of a group
  bootstrap will be much wider than those of an image bootstrap in both cases.

The split files of this run were in `splits/rejected_links_grid/`, as a record. Design change 2
accepts this split, and the files are in `splits/` now (§13).

## 13. Design change 2: the split of the fourth run is the split of the project

Three splits failed the scene check (§8, §10, §12), and the design asked for a decision. The
decision is design change 2 (2026-11-10): the split of the fourth run is accepted with no change,
the scene check becomes a measurement, and the answer is for test images that have no near image in
the training data. It is not an answer for new cameras. The change is merged into `docs/design.md`
(§3, §4.3 "The accepted split", §4.5, §11, §12), as change 1 was. The predictions (§2), the arms,
the training, the metrics, the two automatic checks and the bootstrap of scene groups stay.

**Why the methods failed.** At one limit, one group holds many cameras, and one camera is in many
groups. So two frames of one camera can be farther apart than frames of two cameras, and no limit
separates them. The invented cameras of the simulation have a gap between the two cases. The real
cameras do not.

**Why not a random image split.** Change 1 recommended it for this case, and so did §12 of these
notes. That was wrong. A random split discards the separation that the fourth run has: its test
images have near copies in the training data (a similarity of 1.0000 in the pairs of the sheets),
and the fourth-run split has none at or above 0.7214.

**Checked before the build.** Each number of the change agrees with the committed files:

| | In the change | From the files |
|---|---|---|
| The first 16 characters of the SHA-256 of the seven files | a table | equal for all seven |
| Scene groups of the train subset | 291 | 291 |
| Images from groups with fewer than 100 images: dataset, train subset, test, val | 6.4%, 5.1%, 23.2%, 65.3% | equal |
| Images from groups with 1,000 images or more | 74.2%, 78.4%, 33.0%, 19.8% | equal |
| Test groups at the cap of 300, and their images | 22 groups, 6,600 of 10,001 | equal |
| The interval of 60 of 100 | about 50% to 69% | 50.2% to 69.1% (Wilson, 95%) |
| Candidates of `test_seen` (the extension) | 58,108 in 186 groups | equal |
| `test_seen` with the cap of 300, before the removal of near copies | 12,152 images, 28 groups at the cap | equal |

**What is built.**

| Change | Where |
|---|---|
| The seven files of the fourth run are in `splits/`, with no change | Moved from `splits/rejected_links_grid/`; `tests/test_splits.py` checks the hashes and the counts |
| The counts of the person | `splits/scene_measurement.json`: the scene pairs (60 of 100, with the count of each sheet), the largest group (5 cameras in 50 images), the random image split (50 of 50) |
| The scope of the answer, in the first screen of the report | `report._scope`: one paragraph before "How to read the numbers" |
| The limits of the change, in the place of the two that go | `report.render`, "Limits"; the count has its interval there |
| The composition table | `report.split_composition` calculates it from `scene_groups.csv` and the split lists; `report._composition_limits` writes it |
| The words of the change in the messages | `scripts/make_split.py` ("[measure] scene measurement"), the notebook, `scripts/scene_split_simulation.py` (docstring) |
| The status, the split bullets and the limits | `README.md`, `splits/README.md` |
| The extension `test_seen` | Design §12, as text. Not built: it comes after the main result |

`scenes.py`, `stats.py`, `runs.py`, the training and the evaluation have no change.

**Decisions inside the change.** Please read these: the first two go past "no change to the logic".

1. **The stages after `data` read the split from the repository.** The notebook gave the Drive folder
   to each script (`--splits-dir`). Two things made that wrong for an accepted split. The report
   reads `scene_measurement.json` from the splits folder, and that file is in the repository only.
   And a new run of stage `data` writes to the Drive folder: with the old default (`EMBEDDING =
   "mean"`) it would put the rejected split of the third run there, and the next stage would train
   on it. Now `SPLITS` is `splits/` of the clone, stage `data` writes to `SPLIT_RUN` on Drive, and
   its closing text says if the files of the run are equal to the accepted split.
2. **`EMBEDDING` is `"grid"` by default** in the notebook: it is the embedding of the accepted split.
3. **`make_split.py --splits-dir` has no default.** The default was `splits`, which is the accepted
   split now. A run with no folder would write over it.
4. **Two limits are joined to limits of the change.** "The caps change the weight of the cameras" is
   in "The splits do not have the composition of the dataset" (with "group" for "camera": a group
   is not a camera). "The 20% rule is a choice" is in "The limit is the result of one rule".
5. **The numbers of the limits come from the files.** The report calculates the table and the groups
   at the cap, and reads the count and the limit. It writes them only if the files are there; with
   no measurement file it makes no claim about a count.
6. **The interval of the count** is the 95% Wilson interval, calculated in the report.
7. **The change document is not in `docs/`.** Its content is in `docs/design.md`, as for change 1.

**What was run.** The tests (154 passed). The report text for the accepted split, with invented
results, to read the first screen and the limits. No stage of the notebook ran again: the split is
the one of §12, and the next stage is `checks`.

**Open.** The label completeness decision of §8 (2.0% for clear objects, about 10.5% by best guess,
against the limit of 5%). It comes before the first training run. Closed in §14.

## 14. The label completeness decision

The label check of design §4.5 had two counts and no rule for which objects count (§8). The
decision of 2026-11-10: **the clear objects decide the check.**

| What is counted | With no box | Rate | |
|---|---|---|---|
| Clear objects: a vehicle or a pedestrian that is fully visible, and that moves or is on the road | 6 | 6 of 303 = 2.0% | Decides the check. The limit is 5%: **passed** |
| All objects at a best guess: also tiny, occluded and blurred vehicles, and vehicles at night that show only their lights | about 35 | 35 of 332 = about 10.5% | For the report. It does not decide the check |

The 100 images have 297 boxes. No pedestrian with no box was found.

**What is built.**

| Change | Where |
|---|---|
| The rule and the result | Design §4.5 (two lines under "Label completeness"), §11 ("Incomplete labels"), and a line in the header |
| The counts of the person | `splits/label_measurement.json`, next to `scene_measurement.json`: the report reads one folder |
| A limit in the report with the two rates | `report._label_limit`: "Some objects have no box" |
| The words of the check | `scripts/prepare_data.py` and the closing text of stage 1 in the notebook: which objects decide the check |
| The status and one limit | `README.md`; a row in `splits/README.md` |

**A decision inside it.** The second count is in the report as a limit, with its rate. The reason: a
detection on an object with no box is a false positive, and about 1 object of 10 at a best guess
has no box. This is the same for each arm, so it lowers each AP and it is not a known advantage of
one arm.

With this, each check of stage 1 has its result: the counts, the label check, the class counts, no
overlap, the group check and the split check passed, and the scene measurement is recorded (§13).
The next stage is `checks`.

## 15. Stages 2 to 4 on MIO-TCD: stage `checks`

Stage `checks` of the notebook at commit `02a5757`, 2026-11-10, on the T4 with fp16. It read the
accepted split from the repository (`/content/truckdet/splits`: grid embedding, limit 0.7214). This
was the first run of the model on MIO-TCD images, and the first training with a data loader,
checkpoints and an evaluation on the GPU. All checks passed.

**Stage 2: the start model on the val split.** The COCO checkpoint with the class heads of design
§5.4 and no training, on the 2,017 val images: AP 0.188, AP50 0.282.

| Class | AP50 | The COCO row of its head |
|---|---|---|
| `bus` | 0.851 | `bus` |
| `car` | 0.768 | `car` |
| `motorcycle` | 0.244 | `motorcycle` |
| `pedestrian` | 0.242 | `person` |
| `bicycle` | 0.218 | `bicycle` |
| `pickup_truck` | 0.212 | `truck` |
| `articulated_truck` | 0.160 | `truck` |
| `single_unit_truck` | 0.117 | `truck` |
| `work_van` | 0.009 | `car` |
| `non-motorized_vehicle` | 0.000 | none |

- The check is "AP50 of `car` and `bus` above zero; an AP near zero shows an error in the box format
  or the resize". 0.77 and 0.85 are far above zero, so the boxes and the resize are correct on the
  real images. The reference of design §2 is 0.20 mAP50 for a COCO YOLOv4 (six merged classes).
- The low classes are as the head rows predict. The three truck classes have the same row, so the
  start model cannot tell them apart. `work_van` has the row of `car` and loses to it.
  `non-motorized_vehicle` has no COCO row.
- Each arm starts from this model (step 0 gives its output exactly), so AP 0.188 is the start of
  each arm on the val split.

**Stage 4: the training loop on 16 images.**

| Arm | Epochs | Loss, first → last | Batches with a loss that is not finite | Steps the fp16 scaler skipped | Check |
|---|---|---|---|---|---|
| F | 200 | 9.551 → 0.233 | 0 | 8 of 800 | AP50 on the 16 images: **1.000** (above 0.9). The resumed run continues |
| H | 20 | 9.551 → 4.649 | 0 | 3 of 80 | The loss decreases |
| D | 20 | 9.551 → 4.229 | 0 | 4 of 80 | The loss decreases |
| B | 20 | 9.551 → 2.742 | 0 | 4 of 80 | The loss decreases |

- **These losses are not a result.** They are training losses on 16 images after 20 epochs. They show
  that each arm trains. They say nothing about P1.
- **The loss of epoch 0 is 9.5513 for each of the four arms.** The scaler skipped the first steps of
  each run (4 of 4 for F; 3 of 4 for H, D and B, whose one step came after the last batch). So the
  loss of epoch 0 is the loss of the start model on the same four batches. Four arms with the same
  value to 4 decimal places: the arms start from one model and see the same images in the same
  order, on the GPU too.
- **The scaler skips are at the start,** as in stage 0 (§6): the scale starts at 65,536 and halves
  after each skipped step. After epoch 2 there are 2 more for F (epochs 6 and 80), 1 for D and 1 for
  B. A full run has 6,000 optimizer steps.

**The resume.** Arm F stopped after epoch 65 and started again from `last.pt`.

- The check passed: the loss of epoch 66 (2.308) is near that of epoch 65 (1.800) and not near that
  of epoch 0 (9.551), and the learning rate goes on (8.01e-4 → 7.94e-4).
- The scaler skipped no step in epoch 66. A new scaler skips the first steps (see epoch 0), so the
  state of the scaler came back from the checkpoint. The tests cannot show this: they run on a CPU,
  where the scaler is off.
- The step of +0.51 at the resume is the largest of the 90 epochs around it (the next largest is
  +0.41; the standard deviation of a step is 0.19). The mean loss of the 10 epochs before the
  resume is 2.08, and that of the 10 epochs after it is 2.04. So there is no lasting change, and one
  resume cannot tell a short disturbance from the noise of an epoch of 4 steps.
- I looked for state that a checkpoint can lose, and found none: the real model has no buffers and
  no batch normalisation (48 `LayerNorm` and 9 `RfDetrLayerNorm` layers), so its parameters are its
  whole state, and a checkpoint of arm F holds all of them. On a CPU the resumed run has exactly
  the weights of the run with no stop (`tests/test_training.py`).
- A disturbance of 2 epochs here is 8 optimizer steps. In a full run that is 8 of 6,000.

**The speed here is not the speed of a run.** An epoch of 16 images is 1 to 2 seconds, and a large
part of it is the start of the loader. The times of stage 0 (§6) stay the estimate: 0.95 hours for
a run of H and 1.98 for a run of F, so about 9 hours for the 6 runs of stage `gate`, plus the
loading of the images and 6 scores of the val split.

**Checked for stage `gate`.** The run left a folder `start_model` in the runs folder on Drive. The
learning-rate choice lists only folders with a run name, so it does not read it
(`runs.list_runs`; `select_lr.py` on such a folder asks for the three rates of H).

## 16. Stage 5 on MIO-TCD: the learning rates of H and F, and the gate

Stage `gate` of the notebook at commit `707caef`, 2026-11-10 to 2026-11-11, on the T4 with fp16, in
one session of 14 hours and 40 minutes. These were the first full training runs: 12 epochs of the
8,000-image train subset, an effective batch of 16 (4 × 4), 6,000 optimizer steps. Each run ended
with the score of its final checkpoint on the val split (2,017 images). All numbers here are for
seed 0 on the val split. They are for the choices and the gate. They are not readings: P1, P2 and
P3 are read on the test split, with three seeds.

**Arm H: heads only.**

| Learning rate | Val AP | Val AP50 | Loss of the last epoch | |
|---|---|---|---|---|
| 1e-4 | 0.3149 | 0.439 | 5.451 | |
| 3e-4 | 0.3386 | 0.472 | 5.342 | |
| **1e-3** | **0.3497** | 0.488 | 5.295 | chosen |
| 3e-3 | 0.3491 | 0.484 | 5.315 | added by the rule |

The best of the three rates of the design was 1e-3, at the end of the list, so the rule of design
§6 added one step, 3e-3. It scored lower, so 1e-3 is the choice. The two are 0.0006 apart; with the
tie rule (equal to 3 decimal places: the lower rate) the choice would be the same.

**Arm F: full fine-tuning.** The heads use the rate of H, 1e-3.

| Learning rate of all other parameters | Val AP | Val AP50 | Loss of the last epoch | |
|---|---|---|---|---|
| 3e-6 | 0.5090 | 0.684 | 3.621 | added by the rule |
| **1e-5** | **0.5213** | 0.701 | 3.254 | chosen |
| 3e-5 | 0.5014 | 0.674 | 3.159 | |
| 1e-4 | 0.3840 | 0.531 | 3.571 | |

The best of the three rates of the design was 1e-5, at the low end, so the rule added 3e-6. It
scored lower, so 1e-5 is the choice. The chosen rate is now inside its list for both arms. For F
this is the answer to a risk of design §11 ("F is not fully tuned"): a lower and a higher rate are
both worse.

**The gate: passes.** AP_F − AP_H on the val split = +0.172 (0.350 → 0.521). The bootstrap standard
error is 0.0143 (400 resamples of the 513 scene groups of the val split), so the difference is 12.0
times its standard error. The design asks for 0.05 AP or more and 5 times or more. The start model
scored 0.188 on the same split (§15).

**The checks of each run.**

- **No skipped batch:** 0 batches with a loss that was not finite, in each of the 8 runs.
- **The schedule:** the learning rate reaches its peak at optimizer step 300 (5% of 6,000) and goes
  down to zero at the end, in each run.
- **The same start:** the loss of the first batch is 9.0488 in each of the 8 runs.
- **The merge difference** is 0 for these runs: H and F have no adapters.
- **The fp16 scaler** skipped 4 of 6,000 optimizer steps in each run of H, and 8 to 10 in each run
  of F. The 4 of H are the same in the four runs: 2 in epoch 0, 1 in epoch 4 and 1 in epoch 8. They
  do not come from the data or the rate. The scaler doubles its scale after 2,000 steps with no
  skip, which is 4 epochs; here the next step is then not finite in fp16, and the scaler skips it
  and halves the scale again. So a run loses one step for each such try. Arms D and B have the same
  number of steps, so expect the same there.

**The time, and why it was 14 hours and 40 minutes.** Before the run I gave "about 9 hours". Two
things made it longer.

1. **8 runs, not 6.** The best rate was at an end of its list for each arm, so each arm had a fourth
   run.
2. **A run is slower than the time of stage 0** (§6). Stage 0 timed the training step on one batch
   that was already on the GPU. A run also reads and resizes the images, and its targets are the
   real boxes.

| Arm | Images per second, stage 0 | Images per second, a run | Hours for one run |
|---|---|---|---|
| H | 28.1 | 21.2 | 1.26 (the estimate was 0.95) |
| F | 13.5 | 11.5 | 2.33 (the estimate was 1.98) |
| D | 21.7 | not run yet | about 1.55 (estimate) |
| B | 15.4 | not run yet | about 2.07 (estimate) |

The 8 runs trained for 14.37 hours. The other 18 minutes are the start of the session, the 8
scores of the val split and the gate. An epoch takes 90 to 105 seconds more than the time of stage
0, for H and for F; the estimates for D and B add the same.

**The time of stage `main`.** 3 runs of D (4.7 hours) and 3 runs of B (6.2 hours), then seeds 1 and 2
of the four arms (2 × 7.2 = 14.4 hours): about 25 hours. A fourth run of D adds 1.6 hours, and a
fourth run of B adds 2.1 hours, so 29 hours at most. The intro of the notebook has these times
now. The rule of design §10 for a total that is too large (a smaller train subset for all arms) is
not used: the 8 runs of H and F would need a new run each.
