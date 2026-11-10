# truckdet

**Where must LoRA go in a detector: the backbone or the decoder?**
A controlled test on truck types in traffic-camera images.

> **Status:** the design is complete and the build is in progress. No results yet.

## The question

With the same number of adapter parameters, does LoRA recover more of the
full fine-tuning gain in the backbone or in the decoder?

## Where the question comes from

This project follows an earlier project of mine, `detr-lora`.

In that project I adapted a pre-trained DETR (ResNet-50) to four LVIS truck
classes: `truck`, `pickup_truck`, `trailer_truck` and `fire_engine`. I wrote
LoRA from scratch (rank 8, on the attention query and value projections,
mergeable into the base weights) and the DETR set loss with Hungarian
matching. The trainable part was 0.68% of the model.

The project failed in a specific way:

- **The detector found the trucks but gave the wrong label.** It labelled 78%
  of the trailers and 95% of the fire engines as generic `truck`.
- **LoRA did not help measurably.** It added +0.017 AP to heads-only training,
  with a 95% interval of [−0.014, +0.051].
- **Better features did help.** I kept DETR's boxes and let a
  logistic-regression head on frozen CLIP ViT-L/14 features decide the class.
  AP went from 0.40 to 0.52.

My reading was that the features limited the detector, and that my adapters
were in the wrong position. They came after a frozen ResNet-50, where they
could not change the features.

I could not test that reading on that data:

1. LVIS checked most images for one truck type only, so the labels were
   incomplete.
2. The class `truck` contained its subtypes.
3. The validation set had 258 images and 22 fire-engine boxes.

## Why this project

truckdet tests the reading directly.

- **A model with two positions for the same adapter.** RF-DETR has a
  transformer backbone (DINOv2) and a transformer decoder.
- **Data with clean labels.** MIO-TCD has 110,000 labelled traffic-camera
  images and 11 classes that do not contain each other. They include
  `pickup_truck`, `articulated_truck`, `single_unit_truck` and `work_van`.
- **A test split that can show a difference.** It has about 10,000 images,
  against 258 in the earlier project.

The result of this project is the answer to the question, with its
uncertainty. It is not a leaderboard number.

## The experiment

Four training configurations ("arms") start from the same pre-trained model:

| Arm | Trainable | Adapter parameters |
|---|---|---|
| H | Prediction heads only | 0 |
| D | Heads, and LoRA in the decoder (rank 33, 9 projections) | 152,064 |
| B | Heads, and LoRA in the backbone (rank 9, 22 projections) | 152,064 |
| F | All parameters (full fine-tuning) | — |

The *recovered share* of an arm X is (AP_X − AP_H) / (AP_F − AP_H).

I wrote three predictions before any training run:

| | Prediction | Confirmed if |
|---|---|---|
| P1 | LoRA in the backbone is better than LoRA in the decoder. | The 95% interval of AP_B − AP_D is above zero. |
| P2 | The advantage of the backbone is in the labels, not in the boxes. | Label accuracy on the truck classes is higher for B, and the share of localized boxes is not different. |
| P3 | LoRA in the backbone recovers most of the full fine-tuning gain. | The 95% interval of the recovered share of B is above 2/3. |

A gate comes first. If heads-only and full fine-tuning are less than 0.05 AP
apart on the validation split, the data cannot separate the arms, and the
project stops there.

## How the comparison stays fair

- **Equal adapter budget.** Arms B and D have exactly the same number of LoRA
  parameters.
- **No dead adapters.** The detector does not use the last backbone layer, so
  LoRA skips it.
- **A learning rate for each arm,** chosen on the validation split.
- **Three seeds for each arm,** and a paired bootstrap on the test images.
- **A split by scene.** Frames from one camera show the same background. The
  dataset has no camera ID, so I estimate scene groups from image embeddings
  and keep each group in one split.
- **Ignore regions.** The class `motorized_vehicle` marks vehicles that are
  too small to identify. A specific vehicle class on such a box is not a true
  positive and not a false positive.
- **The test split is scored one time** for each final model.

## Results

*To come.* The report will give:

- AP for each arm and seed, with 95% intervals
- the recovered share of arms B and D
- the outcome of P1, P2 and P3, in the words "confirmed", "refuted" or
  "not shown"
- the error split (boxes and labels) and the confusion table of the truck
  classes

## Written by me, and borrowed

| Written by me | Borrowed |
|---|---|
| LoRA layer, injection and merging (from `detr-lora`) | RF-DETR model, weights and loss (`transformers`) |
| Dataset converter and scene split | COCO AP code (`pycocotools`) |
| Arm definitions and parameter accounting | PyTorch |
| Training loop | |
| Evaluation with ignore regions, error split, bootstrap statistics | |

## Limits

- One model (RF-DETR base) and one dataset. The answer is not a general law.
- Full fine-tuning uses a simple recipe. If it is under-tuned, the recovered
  share of LoRA is too high.
- The scene groups are an estimate.
- Equal parameter counts are not equal capacity. Rank 33 is a larger share of
  a decoder matrix than rank 9 is of a backbone matrix.

## Data and licence

MIO-TCD is available from tcd.miovision.com under CC BY-NC-SA 4.0. This
repository contains code and lists of image IDs. It contains no images.