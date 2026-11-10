"""Stage 1, part 2: scene groups and the split files (design §4.3), with the checks of §4.5.

    python scripts/make_split.py --data-root /content/data/MIO-TCD-Localization --splits-dir splits \\
        --sheets-dir /content/checks

Writes splits/{test,val,train_pool,train}.txt, scene_groups.csv and split_meta.json. Commit them. The scene
check needs eyes: for 200 random test images the sheets show the nearest train-pool image. Accept the split
if 10 pairs or fewer show the same camera view, and do the same count on the sheets of the random image
split, for the report.

Class counts (design §4.5): if a specific class has fewer than 100 boxes in the test split or fewer than 30
in the val split at the sizes of the design, the split takes more whole groups, in the same random order,
until each class has its minimum.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from truckdet.data import scenes
from truckdet.data.miotcd import CLASSES, boxes_per_class, load_annotations
from truckdet.runs import CHECK_FAILED
from truckdet.sheets import pair_sheets
from truckdet.training import ANNOTATIONS, IMAGE_DIR, resolve_loaders


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--sheets-dir", type=Path, help="where the scene-check sheets go (not into the repository)")
    parser.add_argument("--embeddings", type=Path, help="cache of the embeddings (default <data-root>/embeddings.npy)")
    parser.add_argument("--groups-csv", type=Path,
                        help="image_id,scene_group: use these groups and do not cluster (if the IDs show the camera)")
    parser.add_argument("--model-name", default="Roboflow/rf-detr-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--groups", type=int, default=scenes.N_GROUPS)
    parser.add_argument("--test-images", type=int, default=scenes.TEST_IMAGES)
    parser.add_argument("--val-images", type=int, default=scenes.VAL_IMAGES)
    parser.add_argument("--train-images", type=int, default=scenes.TRAIN_IMAGES)
    parser.add_argument("--seed", type=int, default=scenes.SPLIT_SEED)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    coco = load_annotations(args.data_root / ANNOTATIONS)
    images = sorted(coco["images"], key=lambda im: im["miotcd_id"])
    miotcd_ids = [im["miotcd_id"] for im in images]
    coco_ids = np.array([im["id"] for im in images])
    image_dir = args.data_root / IMAGE_DIR

    cache = args.embeddings or args.data_root / "embeddings.npy"
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(images):
        embeddings = np.load(cache)
        print(f"[scenes] embeddings from {cache}")
    else:
        load_model, load_processor = resolve_loaders()
        processor = load_processor(args.model_name)
        backbone = load_model(args.model_name).model.backbone.backbone
        embeddings = scenes.embed_images([image_dir / im["file_name"] for im in images], backbone, processor.image_mean,
                                         processor.image_std, args.device, args.batch_size, args.num_workers)
        np.save(cache, embeddings)
    similarity = scenes.neighbour_similarity(embeddings, args.seed)
    print(f"[scenes] mean cosine similarity: images next to each other in ID order {similarity['adjacent_ids']:.3f}, "
          f"random pairs {similarity['random_pairs']:.3f}. If the first is much higher, the IDs show the camera: "
          "make groups from runs of IDs and pass them with --groups-csv.")

    if args.groups_csv:
        by_id = scenes.read_groups(args.groups_csv)
        groups = np.array([by_id[i] for i in miotcd_ids], dtype=np.int64)
        source = f"groups from {args.groups_csv.name}"
    else:
        groups = scenes.cluster_scenes(embeddings, args.groups, args.seed)
        source = f"k-means, k = {args.groups}, on unit-length mean features of the frozen backbone at {scenes.EMBED_SIZE} px"
    def ids_of(split: dict) -> dict:
        return {name: coco_ids[positions].tolist() for name, positions in split.items()}

    def sizes_of(split: dict) -> str:
        return ", ".join(f"{name} {len(split[name]):,} images in {len(np.unique(groups[split[name]]))} groups"
                         for name in ("test", "val"))

    # The split at the sizes of the design. If a class has too few boxes in it, the design says to make the
    # splits larger: the same groups in the same order, and more of them, until each class has its minimum.
    splits = scenes.split_by_group(groups, args.test_images, args.val_images, args.seed)
    short = scenes.class_count_errors(coco, ids_of(splits))
    made_larger = None
    if short:
        print(f"[check] class counts at the sizes of the design ({sizes_of(splits)}):")
        for line in short:
            print("  - " + line)
        larger = scenes.split_by_group(groups, args.test_images, args.val_images, args.seed,
                                       class_boxes=scenes.class_box_counts(coco, coco_ids),
                                       min_boxes=(scenes.MIN_TEST_BOXES, scenes.MIN_VAL_BOXES))
        if not scenes.class_count_errors(coco, ids_of(larger)) and len(larger["train_pool"]) >= args.train_images:
            made_larger = {"images_at_the_sizes_of_the_design": {name: len(splits[name]) for name in ("test", "val")},
                           "short_at_the_sizes_of_the_design": short}
            splits = larger
            print(f"[check] the splits are made larger (design §4.5), with whole groups in the same random order, "
                  f"until each specific class has {scenes.MIN_TEST_BOXES} boxes in test and {scenes.MIN_VAL_BOXES} "
                  f"in val: {sizes_of(splits)}")
        else:
            print("[check] the splits are NOT made larger: this dataset cannot give each class its minimum and "
                  f"leave {args.train_images:,} images for training")
    splits["train"] = scenes.train_subset(splits["train_pool"], args.train_images, args.seed)

    errors = scenes.overlap_errors(splits, groups)
    by_split = ids_of(splits)
    errors += scenes.class_count_errors(coco, by_split)
    if len(splits["train_pool"]) < args.train_images:
        errors.append(f"train pool: {len(splits['train_pool']):,} images, {args.train_images:,} are needed for the "
                      "train subset")
    counts = {name: boxes_per_class(coco, ids) for name, ids in by_split.items()}
    print("| Class | " + " | ".join(by_split) + " |\n|---|" + "---|" * len(by_split))
    for c in CLASSES:
        print(f"| {c} | " + " | ".join(f"{counts[name][c]:,}" for name in by_split) + " |")
    print("| **images** | " + " | ".join(f"{len(ids):,}" for ids in by_split.values()) + " |")
    print(f"[check] scene groups: {len(np.unique(groups))} ({source}); test {len(np.unique(groups[splits['test']]))}, "
          f"val {len(np.unique(groups[splits['val']]))}, train pool {len(np.unique(groups[splits['train_pool']]))}")
    print("[check] no overlap and class counts: " + ("passed" if not errors else "FAILED"))
    for line in errors:
        print("  - " + line)

    if args.sheets_dir:
        ids = np.array(miotcd_ids)
        for prefix, split in (("scene_split", splits), ("random_split", scenes.random_split(
                len(images), len(splits["test"]), len(splits["val"]), args.seed))):
            pairs = scenes.scene_check_pairs(embeddings, split, args.pairs, args.seed)
            sheets = pair_sheets([(ids[a], ids[b], s) for a, b, s in pairs], image_dir, args.sheets_dir, prefix)
            mean = float(np.mean([s for _, _, s in pairs]))
            print(f"[check] scene check, {prefix}: {len(pairs)} pairs on {len(sheets)} sheets ({args.sheets_dir}/"
                  f"{prefix}_*.jpg), mean similarity {mean:.3f}. Count the pairs that show the same camera view.")
        print("[check] accept the scene split if 10 pairs or fewer show the same camera view.")

    meta = {
        "seed": args.seed, "groups": int(len(np.unique(groups))), "group_source": source, "model": args.model_name,
        "images": {name: len(ids) for name, ids in by_split.items()}, "boxes_per_class": counts,
        "neighbour_similarity": similarity, "requested": {"test": args.test_images, "val": args.val_images,
                                                           "train": args.train_images},
        "class_minimums": {"test": scenes.MIN_TEST_BOXES, "val": scenes.MIN_VAL_BOXES},
        "made_larger_for_class_counts": made_larger,
    }
    scenes.write_split_files(args.splits_dir, miotcd_ids, groups, splits, meta)
    print(f"[scenes] wrote the split files to {args.splits_dir}: " + json.dumps(meta["images"]))
    raise SystemExit(CHECK_FAILED if errors else 0)


if __name__ == "__main__":
    main()
