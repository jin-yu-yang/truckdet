# LoRA position in RF-DETR: backbone against decoder, on MIO-TCD — design

- **Date:** 2026-11-10
- **Status:** ready to build.
- **Changed:** 2026-11-10, change 1: scene groups by links, a split with a cap for each group, and a bootstrap of scene groups (§4.3, §4.5, §7.3, §11). The first method, k-means, failed the scene check on MIO-TCD: `docs/build_notes.md` §8. All other sections are as they were.
- **Changed:** 2026-11-10, change 2: the split of the fourth run on MIO-TCD (links, the second embedding) is the split of the project. The scene check is a measurement now, not a pass rule. The answer is for test images that have no near image in the training data; it is not an answer for new cameras (§3, §4.3, §4.5, §11, §12). Three splits failed the scene check: `docs/build_notes.md` §8, §10 and §12. The predictions and readings (§2), the arms, the training, the metrics, the two automatic checks and the bootstrap of scene groups (§7.3) are as they were.
- **Builds on:** the truck project in `~/detr-lora`. See the README sections "After the fourth round" and "Two-stage classification", `detr-lora/results/clip_second_stage.md`, and the code in `detr-lora/src/detr_lora/model/lora.py`, `detr-lora/src/detr_lora/diagnostics.py` and `detr-lora/src/detr_lora/separation.py`.
- **Home:** a new repository. This file is here because the question comes from this repository's results. Move it when the new repository exists.

## 1. Why

The truck project ended with a claim that it could not test.

- LoRA on the DETR transformer, after a frozen ResNet-50, added +0.017 AP [−0.014, +0.051] to heads-only training.
- A head on frozen CLIP features, on the same boxes, added +0.123 AP [+0.069, +0.148] (0.397 → 0.519).
- The reading was: the features limit the detector, and the adapters were in the wrong position.

That reading is a hypothesis about position. The truck data could not test it. The labels were incomplete, `truck` contained its subtypes, and the validation set had 258 images.

This project tests the hypothesis directly:

- RF-DETR has a transformer backbone and a transformer decoder. The same LoRA layer fits in each position.
- MIO-TCD has 110,000 labelled traffic-camera images and 11 classes that do not contain each other. Three of them are truck types, and one is the work van.

The deliverable is the answer to one question, with its uncertainty. It is not a leaderboard number.

## 2. Question, predictions and readings

**Question.** With the same number of adapter parameters, does LoRA recover more of the full fine-tuning gain in the backbone or in the decoder?

**Terms.**

- An *arm* is one training configuration (§5.2): H (heads only), D (LoRA in the decoder), B (LoRA in the backbone), F (full fine-tuning).
- *AP* is the primary metric of §7.1 on the test split, as the mean of three seeds.
- The *recovered share* of an arm X is R_X = (AP_X − AP_H) / (AP_F − AP_H).
- An *interval* is the 95% paired bootstrap interval of §7.3.

**Predictions.** They are written before any training run. Do not change them after the first result.

| | Prediction | Confirmed if |
|---|---|---|
| P1 | LoRA in the backbone is better than LoRA in the decoder. | The interval of AP_B − AP_D is above zero. |
| P2 | The advantage of the backbone is in the labels, not in the boxes. | For the truck family (§7.2), the interval of the label accuracy difference B − D is above zero, and the interval of the localized share difference B − D includes zero. |
| P3 | LoRA in the backbone recovers most of the full fine-tuning gain. | The interval of R_B is above 2/3. |

A prediction is refuted if its interval is on the opposite side (below zero for P1 and P2, below 2/3 for P3). All other results are "not shown". Each outcome goes in the report, in these words.

**Gate.** The question has no answer if H and F are too near. After the H and F runs with seed 0 (stage 5), calculate AP_F − AP_H on the val split. Continue only if the difference is at least 0.05 AP and at least 5 times its bootstrap standard error. If not, stop, and report that the data cannot separate the arms.

There is a reason to expect a large difference. A published YOLOv4 model that was trained on COCO scored mAP50 0.20 on MIO-TCD. The same model scored 0.86 after training on MIO-TCD. Both numbers are for six merged classes, from the paper "Near real-time social distance estimation in London" (2022).

## 3. Decisions

1. **Dataset:** the labelled part of MIO-TCD localization. The official test images have no public labels, so the project makes its own split.
2. **Split by scene group** (§4.3): images that are near in appearance go to one split. A scene group is not a camera. A random image split puts near copies of the test images in the training data; this split does not.
3. **`motorized_vehicle`** is a class in training and an ignore region in evaluation (§4.4).
4. **Model:** `RfDetrForObjectDetection` from `transformers==5.20.1` (the version that the truck project pins), checkpoint `Roboflow/rf-detr-base`.
5. **Borrowed and written** (§8). The model, its weights and its loss come from the library. The truck project already shows a hand-written matcher and set loss; a second copy for a second detector adds little. The new work is the experiment.
6. **Equal adapter budget:** 152,064 LoRA parameters in arm B and in arm D (§5.3).
7. **A learning rate for each arm,** chosen on the val split (§6). A comparison with one common learning rate is not fair to all arms.
8. **Three seeds for each arm.** The test split is scored one time for each final checkpoint. All choices use the val split.
9. **Fixed length, final checkpoint.** No early stopping, no weight averaging (EMA), no multi-scale training, no layer-wise learning-rate decay. The recipe is the same for all arms.
10. **Predictions are saved.** All analysis runs on a CPU from the saved predictions, as in the truck project.

## 4. Data

### 4.1 Source

- Download: `MIO-TCD-Localization.tar` from `tcd.miovision.com`. Licence: CC BY-NC-SA 4.0.
- Labelled part: `train/*.jpg` and `gt_train.csv`. Reference counts: 110,000 images, 351,549 boxes, 20,299 images with a `motorized_vehicle` box. The first is from the dataset's authors. The other two are from the notes of another user of the dataset.
- `gt_train.csv` has no header. Columns: `image_id, class, xmin, ymin, xmax, ymax` in pixels. Keep the leading zeros of the image ID.
- Classes, in this order: `articulated_truck`, `bicycle`, `bus`, `car`, `motorcycle`, `motorized_vehicle`, `non-motorized_vehicle`, `pedestrian`, `pickup_truck`, `single_unit_truck`, `work_van`.
- The *specific classes* are the 10 classes other than `motorized_vehicle`. The *truck family* is `articulated_truck`, `pickup_truck`, `single_unit_truck` and `work_van`.

### 4.2 Conversion

Convert the CSV to one COCO-format JSON. Read image sizes from the JPEG headers. Clip boxes to the image. Drop boxes with zero area, and record the count.

### 4.3 Scene groups and split

The CSV has no camera ID. The project groups the images that are near in appearance, and gives each group to one split. A *scene group* is a connected set of images that are linked by appearance. It is not a camera: on MIO-TCD no limit of similarity separates the cameras ("The accepted split", below).

The split has one guarantee: no test image has a near image in the train pool. The method makes "near images are in one group" true by construction.

**Embedding.** One embedding for each labelled image: the mean of the last feature map of the frozen backbone, at 224 px, with unit length. *Similarity* is the cosine similarity, and *distance* is 1 − similarity.

**Scene groups for a limit.** For a limit L, link each pair of images with a similarity of L or more. A *scene group* is a connected set of this graph. An image with no link is a group of one image. The groups are a function of the embeddings and L only. They do not depend on a seed or on the implementation.

**Candidate limits.**

- d_near is the median, for all images, of the distance to the nearest image.
- d_rand is the median distance of 110,000 random pairs.
- The 17 candidates have distances at equal steps of log distance from d_near to d_rand: L_k = 1 − d_near × (d_rand / d_near)^(k/16), for k = 0 to 16.

An example, with numbers near those of the first run on MIO-TCD: for a nearest-image similarity of 0.996 and a random-pair similarity of 0.914, the candidates go from 0.996 to 0.914, and each step multiplies the distance by 1.21.

**Over-linked.** A limit is *over-linked* if one of these is true:

- the largest group has more than 20% of the images;
- the split (below) is not possible with its groups.

**The search.** An automatic search sets the limit. Go from the highest candidate to the lowest. Stop at the first over-linked limit. The chosen limit is the candidate before it.

- A lower limit can only join groups, so the search has one stop point.
- The lowest limit that is not over-linked has the most links. It gives a camera the smallest chance to be in two groups.
- If the first candidate is over-linked, there is no limit: go to "If no limit passes".
- The search needs no work by a person. The work of a person is the scene measurement of §4.5.

Record one line for each candidate in `split_meta.json`: the limit, the number of groups, the share of the largest group, and "over-linked" with its cause.

**Implementation.** The implementation is free. The group check of §4.5 decides if it is correct. One method that worked in the test below:

1. Calculate the 100 nearest images of each image, one time.
2. For a limit, take the links of these lists and calculate the connected sets (`scipy.sparse.csgraph.connected_components`).
3. Find each image whose most similar image outside its group is at or above the limit. Add that link. Do step 2 again. Repeat until there is no such image.

Step 3 is necessary. The lists do not hold all links of an image in a large group. One pass of step 3 compares all pairs of images; do it on the GPU, in blocks of rows.

**The split.**

1. Put the groups in a random order (fixed seed).
2. **Test:** groups go to the test split until it has 10,000 images or more, and each specific class has 100 boxes or more. A group gives at most 300 images, drawn at random (fixed seed).
3. **Val:** the next groups go to the val split until it has 2,000 images or more, and each specific class has 30 boxes or more. A group gives at most 100 images.
4. **Train pool:** all images of all other groups.
5. **Not used:** the images of a test or val group that are over its cap. They are in no split. Write them to `unused.txt`.
6. **Train subset:** 8,000 random images of the train pool (fixed seed). It is the same for all runs. §10 gives the rule to change this number.

Count images and boxes on the images that a split uses, not on the full groups. With the caps, no group can dominate a split.

The caps give the minimum number of groups: 34 in the test split (10,000 / 300) and 20 in the val split (2,000 / 100). §7.3 needs these.

The split is *not possible* if the groups end before the val split is complete, or if the train pool has fewer than 8,000 images.

**If no limit passes.** A split fails if the search finds no limit. Until change 2 a split also failed if the scene check failed at the chosen limit: the scene check was a pass rule then (10 pairs of 200 or fewer with the same camera view). Do not try other limits of the same embedding: a higher limit cuts more cameras, and a lower limit is over-linked.

*Step 1: a second embedding, one time.* It keeps the position of things in the picture.

- For each image, take the last feature map of the backbone at 224 px (16 × 16 positions, 384 channels).
- Calculate the mean in each cell of a 4 × 4 grid, and join the 16 cells: 6,144 numbers.
- Subtract the mean of all images. Give each row unit length.
- Do the groups, the search, the split and the checks again.

The mean feature tells what is in a picture. The grid also tells where it is. Two cameras on similar roads show the same things in other positions. It is 2.7 GB as float32; float16 is sufficient for storage. On MIO-TCD the first embedding failed the scene check, so the second was used. It separates the cameras better than the first, but not fully ("The accepted split", below).

*Step 2: decision (change 2).* Change 1 gave two alternatives for a second embedding that fails the scene check: a random image split, or a stop. They are gone. The split of the second embedding is accepted with its scene measurement: see "The accepted split", below. No third embedding and no other limit is tried.

If the file order or the image IDs show the camera, use that as the group and do not link.

The splits are files of image IDs. Commit them. Do not commit images.

**The accepted split.** Three splits failed the scene check on MIO-TCD (`docs/build_notes.md` §8, §10 and §12):

| Split | Nearest train-pool image shows the same camera | Cameras in 50 images of the largest group |
|---|---|---|
| K-means, mean embedding | 125 of 125 pairs | not looked at |
| Links, mean embedding, limit 0.9913 | 71 of 100, and 2 not clear | 14 |
| Links, grid embedding, limit 0.7214 (the fourth run) | 60 of 100 | 5 |
| Random image split | 50 of 50 | |

*No limit separates the cameras.* The looks of the third and fourth runs show two facts at one limit: one group holds many cameras, and one camera is in many groups. So two frames of one camera can be farther apart than frames of two cameras. This is true in the two embeddings. Appearance cannot give a camera split of this dataset. The invented cameras of the test below have a gap between the two cases; the real cameras do not.

*A random image split is not the answer.* Change 1 recommended it for this case. That was wrong: a random split discards the separation that the fourth run has.

| | Random image split | The split of the fourth run |
|---|---|---|
| Highest similarity of a test image to a train-pool image | 1.0000 (in the 200 pairs of the sheets) | 0.7212 (all test images) |
| Mean similarity of a test image to its nearest train-pool image (the 200 pairs of the sheets) | 0.885 | 0.554 |
| Scene groups in the test split, for the bootstrap | none of its own | 1,096 |
| A measure of independent units in the test split: the square of the image count, divided by the sum of the squares of the group counts (for the random split, with the groups of the fourth run) | 30 | 46 |

The split of the fourth run is not a camera split. It is the best separation that the project has, and its guarantee is exact: no test image has a near image in the training data.

*Decisions.*

1. The split of the project is the split of the fourth run, with no change. Do not make a split again.
2. The work on a camera split stops. No third embedding, no other limit. Only camera labels made by a person can give a camera split, and the question of the project does not need one.
3. The scene check becomes a measurement (§4.5). The count of the fourth run, 60 of 100, goes in the report. It is not a pass rule. The group check and the split check stay pass rules.
4. The scope of the answer changes (below).
5. The term "scene group" stays. A scene group is a connected set of images that are linked by appearance. It is not a camera.
6. The unit of the bootstrap is the scene group of this split (§7.3).

*The split.* The seven files are in `splits/`, as `scripts/make_split.py` wrote them in the fourth run. `splits/scene_measurement.json` has the counts of the person.

| | Images | Scene groups |
|---|---|---|
| test | 10,001 | 1,096 |
| val | 2,017 | 513 |
| train pool | 67,785 | 1,452 |
| train subset | 8,000 | 291 |
| not used | 30,197 | |

- Embedding: grid. Limit: 0.7214082. Caps: 300 (test) and 100 (val). Seed: 0.
- Group check: 0 failures. Split check: 0 failures. The highest similarity of a test image to a train-pool image is 0.7212.
- The smallest classes: `motorcycle` 148 and `non-motorized_vehicle` 257 boxes in test; `motorcycle` 45 and `non-motorized_vehicle` 48 in val.

*The scope of the answer.* Use this text, or text with the same content, in the README and the report:

> The test images have no near image in the training data: no training image has a similarity of 0.7214 or more to a test image. The cameras are not separated. For 60 of 100 test images, the nearest training-pool image shows the same camera in other conditions. So the answer is for images that are different from the training images. It is not an answer for new cameras.

The count has an interval of about 50% to 69% (100 pairs). It is a lower limit: a camera can be in the train pool and not be the nearest image.

**The first method, and why it failed.** The first version of this section clustered the embeddings with k-means, k = 1,000, and gave whole clusters to a split. On MIO-TCD it failed the scene check (`docs/build_notes.md` §8):

- In 125 of 125 pairs, the nearest train-pool image of a test image showed the same camera.
- The mean similarity of such a pair was 0.993 for the scene split and 0.996 for a random split. The scene split removed almost nothing.

These facts show two things. The embedding is good at short range: the nearest image is the same camera each time. K-means does not use that: it cuts the space into 1,000 cells, and a cell border goes through the frames of a camera and puts near images in two groups.

**What was tested, and what was not.** The method was tested on synthetic data, with the code of the project: `scripts/scene_split_simulation.py`. It makes 150 invented cameras with 24,171 frames. Each frame is a camera centre, a day-to-night change, and noise. Four cases go from "cameras well apart" (A) to "not separable" (D). The true camera takes the place of the person in the scene check.

| Case | K-means: scene check | Chosen limit | Groups (150 cameras) | Scene check | Failures of the split check | A limit from a count by a person |
|---|---|---|---|---|---|---|
| A | 46% same camera | 0.9689 | 148 | 0% | 0 | 150 groups |
| B | 59% | 0.9837 | 100 | 0% | 0 | 2 groups, largest 100% |
| C | 82% | 0.9954 | 150 | 0% | 0 | 1 group |
| D | 48% | none: the first limit is over-linked | | | | 1 group |

- K-means fails the scene check in each case, as it did on MIO-TCD.
- The search finds a limit that passes in A, B and C.
- In D the search finds no limit. The failure signal works.
- In C, one step higher fails the scene check (8%), and one step lower is over-linked. The interval of good limits can be one step wide.
- Step 3 of the implementation added links in 5 of the 35 limits.
- A smaller copy of case D (40 cameras, in `tests/test_scripts.py`) shows the second failure signal: the search accepts its first candidate, the two automatic checks pass, and the scene check finds the same camera in 85% of the pairs. The scene check by a person is necessary.
- The last column is a rule that this design does not use: a person labels 25 pairs in each bin of similarity, and the limit is the similarity below which 1 pair of 25 or fewer shows the same camera. It set the limit too low in three of the four cases.

Not tested: MIO-TCD. The camera model is invented. The test shows that the mechanics and the failure signals work. It does not show that the cameras of MIO-TCD are separable in the real embedding. Only the scene check on the real images shows that, and it showed that they are not ("The accepted split", above).

### 4.4 `motorized_vehicle`

The dataset uses this class for vehicles that are too small to identify. The official metric does not penalise a specific vehicle class on such a box.

- **Training:** `motorized_vehicle` is class 6 of 11, with no special rule.
- **Evaluation:** for each `motorized_vehicle` box, add a copy with `iscrowd = 1` under each of the seven motorized specific classes: `articulated_truck`, `bus`, `car`, `motorcycle`, `pickup_truck`, `single_unit_truck`, `work_van`. `pycocotools` then ignores a detection of these classes on that box: it is not a true positive and not a false positive.
- Confirm the list of seven against `localization_evaluation.py` in the dataset's code archive.

### 4.5 Checks before any training

- **Counts:** the converter gives the three reference counts of §4.1. Find the cause of each difference.
- **Label completeness:** draw 100 random images with their boxes. Count the vehicles and pedestrians that have no box. If more than 5% have no box, the labels are not complete. Record the rate and stop: this was the first cause of failure in the truck project.
- **Class counts:** each specific class has 100 boxes or more in the test split and 30 or more in the val split. The split of §4.3 takes groups until this is true.
- **No overlap:** no image and no scene group is in two splits.
- **Group check (automatic):** for each image, find its most similar image outside its group. No such similarity is at or above the chosen limit. The count of failures must be 0.
- **Split check (automatic):**
  - No test image and no val image has a train-pool image at or above the limit.
  - No test image has a val image at or above the limit.
  - The test split has 34 groups or more, and the val split has 20 or more.

  The first two lines follow from the group check. They are a guard for the split code.
- **Scene measurement (a person):** for 200 random test images, find the nearest train-pool image by cosine similarity. Look at the pairs, and count the pairs that show the same camera view. Record the count. It goes in the report. Do the same count for a random image split, for the report. After the split check, each of these pairs has a similarity below the limit, so the count tells how many test images have their camera in the train pool at a lower similarity. This was a pass rule until change 2 ("the scene check": 10 pairs of 200 or fewer), and three splits failed it. It is a measurement now (§4.3, "The accepted split"): for the accepted split, 60 of 100 pairs show the same camera view.

The group check, the split check and the scene measurement come in this order. The group check and the split check are pass rules.

## 5. Model and arms

### 5.1 Model facts

Measured on the library's default configuration with 11 classes (random weights; `transformers` 5.20.1). The library documents this configuration as the `Roboflow/rf-detr-base` style. Confirm each number on the real checkpoint in stage 0. One difference is probable: an example in the library's documentation shows 200 queries for this checkpoint, not the default 300. That changes the count of the queries only. It does not change the heads or the adapters.

| Part | Module path | Parameters | Notes |
|---|---|---|---|
| Backbone | `model.backbone.backbone` | 22,056,576 | DINOv2-S: 12 layers, width 384, 6 heads, patch 14, 4 × 4 windows. Attention: `encoder.layer.{i}.attention.{q_proj,k_proj,v_proj,o_proj}`. |
| Projector | `model.backbone.projector` | 1,444,864 | Convolutional. |
| Decoder | `model.decoder` | 4,687,168 | 3 layers, width 256. Self-attention: `layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}`. Deformable cross-attention: `layers.{i}.cross_attn.{sampling_offsets,attention_weights,value_proj,output_proj}`. |
| Queries and proposals | `model.query_feat`, `model.reference_point_embed`, `model.enc_output`, `model.enc_output_norm` | 1,875,952 | 300 queries × 13 groups in training. |
| Heads | `class_embed`, `bbox_embed`, `model.enc_out_class_embed.{0–12}`, `model.enc_out_bbox_embed.{0–12}` | 1,896,146 | 13 copies of the proposal heads, one for each group. |
| **Total** | | **31,960,706** | |

Four facts change the design:

- **Backbone layer 11 has no effect on the output.** The detector uses the features after layers 2, 5, 8 and 11 of 12, so the last layer (index 11, about 1.8 M parameters) is calculated and not used. A change to its weights changes no logit. LoRA in that layer is dead weight.
- **Only group 0 is used in inference.** The model has 300 queries and uses `class_embed`, `bbox_embed` and copy 0 of the proposal heads: 270,878 of the 1,896,146 head parameters.
- **`freeze_backbone()` fails in this version** (`AttributeError`: the method reads an attribute that does not exist). Freeze parameters by name.
- **The input side must be a multiple of 56** (patch 14 × 4 windows). 560 px works.

The library calculates the loss when the call includes `labels` (a list of dicts with `class_labels` and `boxes` in normalised centre format). It is `LwDetrForObjectDetectionLoss`: Hungarian matching for each query group, an IoU-aware sigmoid class loss, L1 and GIoU box losses, and the same terms for 2 auxiliary decoder layers and the encoder proposals. It needs `scipy`. It reads the class count from `config.num_labels`.

### 5.2 Arms

| Arm | Trainable | Trainable parameters |
|---|---|---|
| H | The heads of §5.1. All other parameters are frozen. | 1,896,146 |
| D | H, and LoRA in the decoder (§5.3). | 2,048,210 |
| B | H, and LoRA in the backbone (§5.3). | 2,048,210 |
| F | All parameters. | 31,960,706 |

In arms H, D and B the projector, the queries and the proposal layers stay frozen. The adapter position is then the only difference between D and B.

### 5.3 LoRA

Use `LoRALinear`, `inject_lora` and `merge_lora` from the truck project. They work on this model with other names (tested: the output at step 0 is identical, and the merge difference is 1.9e-6).

| Arm | Layers | Projections | Rank | α | Adapter parameters |
|---|---|---|---|---|---|
| B | `q_proj` and `v_proj` in `attention`, backbone layers 0–10 | 22 (384 × 384) | 9 | 18 | 22 × 9 × 768 = **152,064** |
| D | `q_proj` and `v_proj` in `self_attn`, and `value_proj` in `cross_attn`, decoder layers 0–2 | 9 (256 × 256) | 33 | 66 | 9 × 33 × 512 = **152,064** |

- `inject_lora` selects layers by name, so it also selects backbone layer 11. Add a filter that skips it.
- Deformable cross-attention has no query-key product. `value_proj` is its value projection, so it takes the place of `v_proj`.
- The ranks 9 and 33 make the two counts exactly equal. α = 2 × rank in both arms, as in the truck project.
- If the real checkpoint has other sizes, choose integer ranks again so that the counts are equal.
- Equal counts are not equal capacity. Rank 33 is 13% of the full rank of a decoder matrix; rank 9 is 2% of a backbone matrix. This is to the advantage of D, so a result for B is the stronger result.

### 5.4 Class heads

The checkpoint has COCO classes. Make new 11-class layers for `class_embed` and the 13 `enc_out_class_embed` copies. Initialise each row from the COCO row of the same layer, by class name through `config.id2label`:

| MIO-TCD class | COCO row |
|---|---|
| `articulated_truck`, `pickup_truck`, `single_unit_truck` | `truck` |
| `car`, `work_van`, `motorized_vehicle` | `car` |
| `bus`, `bicycle`, `motorcycle` | the same name |
| `pedestrian` | `person` |
| `non-motorized_vehicle` | the library's default initialisation |

Set `config.num_labels = 11`. The box heads keep their weights. All arms start from this same model.

## 6. Training

The same for all arms, unless stated:

- **Input:** the square size, resize and normalisation of the checkpoint's image processor (560 × 560 px expected). **Augmentation:** horizontal flip.
- **Length:** 12 epochs of the 8,000-image train subset. The final checkpoint is the result.
- **Optimiser:** AdamW, weight decay 1e-4, gradient clipping at 0.1, mixed precision. Linear warm-up for the first 5% of the steps, then a cosine decrease to zero.
- **Batch:** an effective batch of 16 images (for example 4 × 4 accumulation steps; stage 0 sets the split by memory).
- **Speed:** in arms H and D, run the backbone and the projector in `torch.no_grad()`. No trainable parameter comes before the decoder.
- **Seed:** sets the initial adapter weights, the data order and the augmentation. Seeds 0, 1 and 2.
- **Checkpoints:** trainable parameters only for H, D and B. Resume after a Colab disconnect, as in the truck project.

**Learning rate, chosen with seed 0 on the val split:**

- H, D, B: one learning rate for all trainable parameters, from {1e-4, 3e-4, 1e-3}.
- F: the heads use the learning rate that H chose. All other parameters use one learning rate from {1e-5, 3e-5, 1e-4}.
- Rule: the highest val AP of the final checkpoint. If two are equal to 3 decimal places, the lower rate.
- If the best rate is at an end of its list, add one step in that direction, one time.
- The seed-0 run with the chosen rate is also the first of the three seeds.

## 7. Evaluation and statistics

### 7.1 Metrics

- **Detections:** the 100 best (query, class) pairs for each image, by sigmoid score (`post_process_object_detection` with `threshold=0`, `top_k=100`). This is the "pairs" reading of the truck project.
- **Primary metric:** COCO AP (IoU 0.50 to 0.95) with `pycocotools`, as the mean of the 10 specific classes, with the ignore regions of §4.4.
- **Also reported:** AP50, AP75, AP for small boxes, AP for each class, the mean AP of the truck family, AP of `motorized_vehicle`, and the 11-class AP with no ignore regions.
- **Reference only:** the public leaderboard shows mAP 0.80 at best. It uses another split, all the training data and another protocol. Do not present the two numbers as a comparison.

### 7.2 Error split: boxes and labels

For P2, each arm gets three more numbers. The first two use `box_table` and `confusion_counts` from `separation.py`.

- **Candidates:** the 50 queries of an image with the highest class score.
- **Localized share:** the share of ground-truth boxes that have a candidate with IoU ≥ 0.5. The candidate with the highest IoU is the *representative query* of the box.
- **Label accuracy:** of the localized boxes, the share whose representative query has the correct top class.
- **Box AP:** AP with all classes merged into one. Each query gives one detection, with its highest class score. This number is in the report and not in a reading.

Calculate them for all specific classes and for the truck family. Report the confusion table of the truck family for each arm.

### 7.3 Uncertainty

- **Paired bootstrap of scene groups:** frames of one camera are not independent, and a bootstrap that resamples images gives intervals that are too narrow. So resample the scene groups of the test split with replacement, 400 times: for a split with G groups, draw G groups. The weight of an image is the number of times its group was drawn. In each resample, score all arms and all seeds on the same images, and take the mean of the seeds for each arm. The interval is the 2.5% to 97.5% range. Reuse `diagnostics.py`; `evaluation.average_precision` and `errors.bootstrap_error_split` take the weights with no change.
- **All intervals and readings use the group bootstrap:** P1, P2, P3 and the recovered share on the test split, and the gate on the val split.
- **For the report:** show the image interval of AP_B − AP_D next to the group interval, one time. The difference shows the reader why the unit is important.
- **Recovered share:** calculate R_X in each resample from the seed means. Its interval comes from the same 400 values.
- **Seeds:** report AP for each seed, and the number of seed pairs that agree in sign with the mean difference (`seed_pairings`).
- **The gate** uses the val split and seed 0 only.

## 8. Units

A suggested layout for the new repository. Tests are CPU-only with synthetic fixtures, as in the truck project.

| Unit | Responsibility | Source |
|---|---|---|
| `data/miotcd.py` | CSV to COCO JSON; counts; class order. | New |
| `data/scenes.py` | Embeddings, neighbour lists, scene groups for a limit, the search, the split with caps, the group check and the split check, split files, nearest-neighbour sheet. | New |
| `data/dataset.py` | Dataset and collate function with the library's image processor. | New |
| `model/lora.py` | `LoRALinear`, `inject_lora` with a layer filter, `merge_lora`, `count_params`. | Truck project, one change |
| `model/arms.py` | Load the checkpoint, make the class heads, freeze by name, add the adapters of an arm, print the parameter table. | New |
| `training.py`, `engine.py` | Optimiser groups, schedule, one epoch, resumable run. | Truck project, adapted |
| `evaluation.py` | Predictions to COCO results; ignore regions; AP tables; box AP. | New |
| `errors.py` | Localized share, label accuracy, confusion table. | From `separation.py` |
| `stats.py` | Paired bootstrap of scene groups, recovered share, the readings of §2 as text. | From `diagnostics.py` |
| `scripts/` | `prepare_data`, `make_split`, `train`, `evaluate`, `analyse`; `scene_split_simulation` (the test of §4.3). | New |
| Library | Model, weights, loss, image processor, `pycocotools`. | Borrowed |

## 9. Build stages and checks

Each stage ends with a check. Do not start a stage before the check of the stage before it passes.

| Stage | Work | Check |
|---|---|---|
| 0. Spike | Load the checkpoint. Print the configuration and the parameter table. Draw detections on 20 images. Measure images per second and peak memory on the T4 for one H step and one F step. | The numbers of §5.1 are confirmed or corrected. A change to `q_proj` of backbone layer 11 changes no logit; a change to each of layers 0–10 does. |
| 1. Data | Converter, scene groups, split files. | The checks of §4.5. |
| 2. Evaluation | AP with ignore regions, box AP, the error split. | Synthetic tests: a `car` detection on a `motorized_vehicle` box changes no AP; the same detection on background, with the highest score of the image set, decreases the `car` AP. The COCO checkpoint with no training scores a low AP50 above zero on `car` and `bus` (reference: 0.20 in §2). An AP near zero shows an error in the box format or the resize. |
| 3. Arms | `arms.py` and the layer filter. | The parameter table equals §5.2. At step 0 each arm gives the output of the start model exactly. Each adapter changes the output when its B matrix is not zero. The merge difference is below 1e-4. |
| 4. Training loop | One configuration file for each arm. | On 16 images, arm F reaches AP50 above 0.9, and the loss of arms H, D and B decreases. A stopped and resumed run continues the loss curve. |
| 5. Gate | Learning-rate runs for H and F (6 runs). | The gate of §2. |
| 6. Main runs | Learning-rate runs for D and B (6 runs). Seeds 1 and 2 for all arms (8 runs). | Each run ends with no skipped step. Each LoRA checkpoint passes the merge check. |
| 7. Analysis | Tables, intervals, the three readings. | The report script gives the same text from the saved predictions on a second run. |
| 8. Report | One figure (recovered share with intervals, by arm), one results table, the error split, sample images, limits. | A reader can find the question, the predictions and their outcomes in the first screen. |

## 10. Compute

- 20 training runs: 12 for the learning rates and 8 for the other seeds. Each run is 96,000 image passes (8,000 images × 12 epochs).
- Time for one run in hours = 96,000 / (images per second) / 3,600. Stage 0 measures the speed. An estimate before measurement: an F run or a B run takes 2 to 3 hours on a T4, and an H run or a D run takes about half of that.
- The gate comes after 6 runs. If it fails, the other 14 runs are not necessary.
- If the total is too large, decrease the train subset first (8,000 → 4,000), then the epochs (12 → 8). Make the same change for all arms. Do not decrease the test split, and do not decrease the learning-rate list of one arm only.

## 11. Risks and limits

- **Incomplete labels.** The dataset describes its boxes as foreground objects. Parked vehicles possibly have no box. The label check of §4.5 measures this before any training.
- **A small difference between H and F.** The gate stops the project early.
- **F is not fully tuned.** With no weight averaging and no layer-wise decay, AP_F is possibly low. Then R_B is too high. The learning-rate choice decreases this risk; the report must state it.
- **The cameras are not separated.** The test images have no near image in the training data: no training image has a similarity of 0.7214 or more to a test image. For 60 of 100 test images, the nearest training-pool image shows the same camera in other conditions. The count has an interval of about 50% to 69%, and it is a lower limit: a camera can be in the train pool and not be the nearest image. So the answer is for images that are different from the training images. It is not an answer for new cameras.
- **The test images were selected with the features of the frozen backbone.** They are far from the training images in those features. Arms H and D keep these features; arms B and F change them. This is possibly to the advantage of B and F.
- **Two limits go in the same direction.** The shared cameras and the selection are both possibly to the advantage of the backbone arms. So a confirmed P1 is the weaker result, and a refuted P1 is the stronger result. The rank limit of §5.3 goes in the opposite direction.
- **The splits do not have the composition of the dataset.** The cause is the caps and the sizes of the groups. A group with many images has no more weight in the test split than a group with 300. The train subset keeps the weights of the train pool.

  | Share of the images that come from | Dataset | Train subset | Test | Val |
  |---|---|---|---|---|
  | groups with fewer than 100 images | 6.4% | 5.1% | 23.2% | 65.3% |
  | groups with 1,000 images or more | 74.2% | 78.4% | 33.0% | 19.8% |

- **The val split is different from the test split.** The val cap of 100 is the cause. The learning-rate choice and the gate use the val split.
- **A few groups give most of the test split.** 22 groups are at the cap of 300 and give 6,600 of the 10,001 images. One group can hold more than one camera: 50 images of the largest group show 5 cameras.
- **The limit is the result of one rule.** The search stops when the largest group has more than 20% of the images, and this rule set 0.7214. The 20% is a choice. No other limit was tried.
- **One model and one dataset.** The answer is for RF-DETR-base on traffic-camera images. It is not a general law.
- **The library port.** Numbers can be different from those of the original `rfdetr` package.
- **Memory.** Training uses 3,900 queries. Stage 0 sets the batch size.
- **Licence.** CC BY-NC-SA 4.0. Publish code and split lists. Read the licence before you publish images.

## 12. Not in scope, and extensions

Not in scope: a new architecture, a hand-written loss, a leaderboard result, other detectors.

Extensions, after the main result and in this order:

1. **The weight change of F as a predictor.** For each adapted matrix, calculate W_F − W_0 and its singular values. Report the share of the energy in the first 9 (backbone) or 33 (decoder) values. Question: is the full fine-tuning change of low rank in the position where LoRA is sufficient? This needs no new training.
2. **Arm BD:** half of the budget in each position.
3. **Early and late backbone layers** at an equal budget (layers 0–4 against 5–10).
4. **Data size:** arms H, B and F with 1,000 and 32,000 training images.

**One more extension: a second test set of seen groups, `test_seen`.** It is optional, and it has no place in the order above. Build it after the main result. It needs no new training, and it does not change the train subset. P1, P2 and P3 are read on the test split only.

- *The question.* Does the difference between the arms change when the test images are near the training images?
- *Candidates.* The train-pool images that are not in the train subset, in a scene group with one training image or more. There are 58,108 in 186 groups.
- *Remove near copies.* A candidate goes if a training image has a similarity to it at or above 0.8980. This is the first candidate limit of the search: the median similarity of an image to its nearest image.
- *Cap.* A group gives at most 300 images, drawn at random (fixed seed), as in the test split. Before the removal of near copies this gives 12,152 images, with 28 groups at the cap.
- *File.* Write the list to `splits/test_seen.txt`. Do not use the set for a choice.
- *One count by a person.* For 50 random images of the set, look at the nearest training image. Count the pairs that show the same moment: the same vehicles in the same positions. Record the count in the report.
- *Scoring.* Each final checkpoint, one time. The bootstrap resamples the scene groups of `test_seen`. Its groups and the groups of the test split are different groups, so resample the two sets independently.
- *Readings, fixed now.* Report AP_B − AP_D on `test_seen`, with its interval. Calculate the change: (AP_B − AP_D on `test_seen`) − (AP_B − AP_D on test). "The difference between the arms depends on nearness" if the interval of the change does not include zero. If not, "not shown".
- *A limit.* The two sets also have different compositions. `test_seen` comes from groups with training images, and most of them are large. A change between the sets can come from the composition and not from nearness.
