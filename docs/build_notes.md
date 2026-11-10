# Build notes

What the build confirmed, corrected and decided. The design is `docs/design.md`; it is not changed.
Each item gives the section of the design and the place in the code.

Date of the build: 2026-11-10. Library: `transformers` 5.20.1. Checkpoint: `Roboflow/rf-detr-base`.

## 1. What has run, and what has not

| | Where it ran | Result |
|---|---|---|
| Stage 0 checks (`scripts/spike.py`), without the timing | Real checkpoint, CPU | All passed (§2 below) |
| Stage 0 checks with the timing (`scripts/spike.py --time`) | Real checkpoint, Colab T4, fp16 | All passed; speed and memory in §6 |
| Stage 1 (`prepare_data.py`, `make_split.py`), first run | MIO-TCD, Colab T4 | The counts and the ignore classes are confirmed; the val split of the design's size has too few boxes of three classes (§7) |
| Stage 1, second run | MIO-TCD, Colab T4 | The class counts pass with a val split of 5,724 images. **The scene check fails:** the scene groups do not separate the cameras (§8). The split is not accepted |
| Tests (`pytest`) | Tiny random RF-DETR and a synthetic MIO-TCD folder, CPU | 132 passed |
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
- **k-means is scikit-learn's**, with one start and a fixed seed.
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

The split of this run is not accepted. Its files are not committed, and no training starts on it.
The design gives no rule for a failed scene check, so the next method is a change to design §4.3.

**COCO detections: looked at.** On the sheet of 20 images, each box of the checkpoint is on its
object, and no class name is wrong.
