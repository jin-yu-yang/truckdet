# Build notes

What the build confirmed, corrected and decided. The design is `docs/design.md`; it is not changed.
Each item gives the section of the design and the place in the code.

Date of the build: 2026-11-10. Library: `transformers` 5.20.1. Checkpoint: `Roboflow/rf-detr-base`.

## 1. What has run, and what has not

| | Where it ran | Result |
|---|---|---|
| Stage 0 checks (`scripts/spike.py`), without the timing | Real checkpoint, CPU | All passed (§2 below) |
| Tests (`pytest`) | Tiny random RF-DETR and a synthetic MIO-TCD folder, CPU | 128 passed |
| Tests of the real checkpoint (`pytest -m slow`) | Real checkpoint, CPU | 5 passed |
| The start model on real images | 60 LVIS truck images written in MIO-TCD's CSV format, CPU | AP50 0.30 with no training (§5) |
| Arm F on 8 real images | The same stand-in images, CPU | See §5 |
| **Not run:** anything on a GPU | | The fp16 path, the speed and the memory are first measured by stage `spike` of the notebook |
| **Not run:** anything on MIO-TCD | | The download, the counts, the embeddings and k-means at 110,000 images first run in stage `data` |

So the first Colab run is the first run of the GPU path and of the real data. The notebook starts
with the two stages that test them.

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

## 5. The check on real images

MIO-TCD is not on the build machine. The 258 val images of the truck project (LVIS) were written
in MIO-TCD's CSV format (`truck` and `fire_engine` → `single_unit_truck`, `trailer_truck` →
`articulated_truck`) to run the real checkpoint through the real code. These numbers are a check of
the code. They are not results.

- The start model, with no training, on 60 of these images: AP 0.25, AP50 0.30
  (`single_unit_truck` 0.46, `pickup_truck` 0.29, `articulated_truck` 0.17), box AP 0.32. The
  resize, the box format and the class rows are correct.
