"""Stage 1, part 2: scene groups and the split files (design §4.3), with the checks of §4.5.

    python scripts/make_split.py --data-root /content/data/MIO-TCD-Localization --splits-dir /content/splits \\
        --embedding grid --sheets-dir /content/checks

A scene group is a connected set of images that are linked at or above a similarity limit, and an automatic
search sets the limit. A group gives at most 300 images to the test split and at most 100 to the val split.
The script prints one line for each limit that the search tries, then the split and its checks:

- no overlap and class counts;
- the group check: no image has an image outside its group at or above the limit;
- the split check: no test or val image has a train-pool image at or above the limit;
- the scene measurement, which needs eyes: for 200 random test images the sheets show the nearest train-pool
  image. Count the pairs that show the same camera view, and record the count. It goes in the report. It is a
  measurement, not a pass rule (design §4.5). Do the same count on the sheets of the random image split.

Writes {test,val,train_pool,train,unused}.txt, scene_groups.csv and split_meta.json to --splits-dir.

The split of the project is in the repository's splits/ folder. It is the output of this script on MIO-TCD
with --embedding grid, accepted with its scene measurement (design §4.3, "The accepted split"). Do not make
that split again: --splits-dir has no default, so give it another folder.

Exit code 3: a check failed, or the search found no limit. With no limit there is no split: the old split
lists are removed, and split_meta.json has the record of the search. Then run the second embedding one
time (--embedding grid). The design has no step after the second embedding (§4.3).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from truckdet.data import scenes
from truckdet.data.miotcd import CLASSES, boxes_per_class, load_annotations
from truckdet.runs import CHECK_FAILED
from truckdet.sheets import group_sheet, pair_sheets
from truckdet.training import ANNOTATIONS, IMAGE_DIR, resolve_loaders

EMBEDDINGS = {
    "mean": ("embeddings.npy", f"the mean of the last feature map of the frozen backbone at {scenes.EMBED_SIZE} px"),
    "grid": ("embeddings_grid.npy", f"the last feature map of the frozen backbone at {scenes.EMBED_SIZE} px, as the mean "
             f"in each cell of a {scenes.GRID} x {scenes.GRID} grid, with the mean of all images subtracted"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", required=True, type=Path,
                        help="where the split files go. Not the repository's splits/: that is the accepted split")
    parser.add_argument("--sheets-dir", type=Path,
                        help="where the sheets of the scene measurement go (not into the repository)")
    parser.add_argument("--embedding", choices=sorted(EMBEDDINGS), default="mean",
                        help="grid: the second embedding of design §4.3. The accepted split of the project uses it")
    parser.add_argument("--embeddings", type=Path, help="cache of the embeddings (default <data-root>/embeddings.npy, "
                                                        "or embeddings_grid.npy for --embedding grid)")
    parser.add_argument("--groups-csv", type=Path,
                        help="image_id,scene_group: use these groups and do not link (if the IDs show the camera)")
    parser.add_argument("--model-name", default="Roboflow/rf-detr-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-images", type=int, default=scenes.TEST_IMAGES)
    parser.add_argument("--val-images", type=int, default=scenes.VAL_IMAGES)
    parser.add_argument("--train-images", type=int, default=scenes.TRAIN_IMAGES)
    parser.add_argument("--test-cap", type=int, default=scenes.TEST_CAP, help="images that one group gives to test")
    parser.add_argument("--val-cap", type=int, default=scenes.VAL_CAP, help="images that one group gives to val")
    parser.add_argument("--seed", type=int, default=scenes.SPLIT_SEED)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    coco = load_annotations(args.data_root / ANNOTATIONS)
    images = sorted(coco["images"], key=lambda im: im["miotcd_id"])
    miotcd_ids = [im["miotcd_id"] for im in images]
    ids = np.array(miotcd_ids)
    coco_ids = np.array([im["id"] for im in images])
    image_dir = args.data_root / IMAGE_DIR
    n = len(images)

    file_name, description = EMBEDDINGS[args.embedding]
    cache = args.embeddings or args.data_root / file_name
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == n:
        raw = np.load(cache)
        print(f"[scenes] embeddings ({args.embedding}) from {cache}")
    else:
        load_model, load_processor = resolve_loaders()
        processor = load_processor(args.model_name)
        backbone = load_model(args.model_name).model.backbone.backbone
        raw = scenes.embed_images([image_dir / im["file_name"] for im in images], backbone, processor.image_mean,
                                  processor.image_std, args.device, args.batch_size, args.num_workers,
                                  grid=scenes.GRID if args.embedding == "grid" else None)
        np.save(cache, raw)
    unit = scenes.as_tensor(scenes.prepare_embeddings(raw, centre=args.embedding == "grid"), args.device)
    del raw
    similarity = scenes.neighbour_similarity(unit, args.seed)
    print(f"[scenes] mean cosine similarity: images next to each other in ID order {similarity['adjacent_ids']:.3f}, "
          f"random pairs {similarity['random_pairs']:.3f}. If the first is much higher, the IDs show the camera: "
          "make groups from runs of IDs and pass them with --groups-csv.")

    class_boxes = scenes.class_box_counts(coco, coco_ids)
    caps = (args.test_cap, args.val_cap)
    min_boxes = (scenes.MIN_TEST_BOXES, scenes.MIN_VAL_BOXES)
    min_groups = (math.ceil(args.test_images / args.test_cap), math.ceil(args.val_images / args.val_cap))

    def try_split(groups: np.ndarray):
        split = scenes.split_with_caps(groups, args.test_images, args.val_images, caps, args.seed, class_boxes, min_boxes)
        return split, scenes.split_problems(split, args.test_images, args.val_images, args.train_images, class_boxes, min_boxes)

    meta = {
        "seed": args.seed, "model": args.model_name, "embedding": args.embedding, "neighbour_similarity": similarity,
        "requested": {"test": args.test_images, "val": args.val_images, "train": args.train_images},
        "caps": {"test": args.test_cap, "val": args.val_cap}, "class_minimums": {"test": min_boxes[0], "val": min_boxes[1]},
        "limit": None, "search": None,
    }
    errors = []
    if args.groups_csv:
        by_id = scenes.read_groups(args.groups_csv)
        groups = scenes.canonical_groups(np.array([by_id[i] for i in miotcd_ids], dtype=np.int64))
        splits, problems = try_split(groups)
        errors += [f"the split is not complete: {line}" for line in problems]
        limit = None
        meta["group_source"] = f"groups from {args.groups_csv.name}"
    else:
        lists = scenes.neighbour_lists(unit)
        d_near, d_rand = scenes.link_distances(unit, lists, args.seed)
        print(f"[search] nearest image: median similarity {1 - d_near:.4f}; random pairs: median similarity "
              f"{1 - d_rand:.4f}. {scenes.CANDIDATES} candidate limits between them, at equal steps of log distance.")

        def log(row: dict) -> None:
            print(f"[search] limit {row['limit']:.4f}: {row['groups']:,} groups, the largest has "
                  f"{row['largest_group_share']:.1%} of the images"
                  + (f"   OVER-LINKED: {row['over_linked']}" if row["over_linked"] else ""), flush=True)

        limits = scenes.candidate_limits(d_near, d_rand)
        result = scenes.search_limit(unit, lists, limits, try_split, log=log)
        chosen = result["chosen"]
        meta["search"] = {"d_near": d_near, "d_rand": d_rand, "neighbours": scenes.NEIGHBOURS,
                          "max_group_share": scenes.MAX_GROUP_SHARE, "candidates": result["candidates"],
                          "chosen_candidate": chosen["candidate"] if chosen else None}
        meta["group_source"] = (f"connected sets of the images that are linked at a cosine similarity at or above "
                                f"the limit; embedding: {description}")
        if chosen is None:
            removed = scenes.remove_split_files(args.splits_dir)
            meta["checks"] = {"passed": False, "errors": ["the search found no limit: the first candidate is over-linked"]}
            scenes.write_meta(args.splits_dir, meta)
            print(f"[check] NO LIMIT: the first candidate is over-linked ({result['candidates'][0]['over_linked']}). "
                  "There is no split.")
            if removed:
                print(f"[scenes] removed the split files of an earlier run from {args.splits_dir}: {', '.join(removed)}")
            print("[check] next (design §4.3, \"If no limit passes\"): " + (
                "run the second embedding one time: --embedding grid." if args.embedding == "mean" else
                "the second embedding gave no limit too. The design has no third embedding and no other limit, so "
                "this method gives no split for these images."))
            raise SystemExit(CHECK_FAILED)
        limit, groups, splits = chosen["limit"], chosen["groups"], chosen["split"]
        meta["limit"] = limit
        print(f"[check] limit: {limit:.4f} (candidate {chosen['candidate'] + 1} of {len(limits)}), the lowest that is "
              "not over-linked")

    sizes = np.bincount(groups)
    largest = float(sizes.max() / n)
    splits["train"] = scenes.train_subset(splits["train_pool"], args.train_images, args.seed)
    groups_in = {name: int(len(np.unique(groups[splits[name]]))) for name in ("test", "val", "train_pool")}
    by_split = {name: coco_ids[splits[name]].tolist() for name in ("test", "val", "train_pool", "train")}
    counts = {name: boxes_per_class(coco, image_ids) for name, image_ids in by_split.items()}
    print("| Class | " + " | ".join(by_split) + " |\n|---|" + "---|" * len(by_split))
    for c in CLASSES:
        print(f"| {c} | " + " | ".join(f"{counts[name][c]:,}" for name in by_split) + " |")
    print("| **images** | " + " | ".join(f"{len(image_ids):,}" for image_ids in by_split.values()) + " |")
    print("| **scene groups** | " + " | ".join(f"{groups_in[name]:,}" if name in groups_in else "" for name in by_split) + " |")
    print(f"[check] scene groups: {len(sizes):,}; the largest has {sizes.max():,} images ({largest:.1%}). A test group "
          f"gives at most {args.test_cap} images and a val group at most {args.val_cap}: {len(splits['unused']):,} "
          "images of these groups are over the cap and are not used.")

    errors += scenes.overlap_errors(splits, groups) + scenes.class_count_errors(coco, by_split)
    if len(splits["train_pool"]) < args.train_images:
        errors.append(f"train pool: {len(splits['train_pool']):,} images, {args.train_images:,} are needed for the "
                      "train subset")
    print("[check] no overlap and class counts: " + ("passed" if not errors else "FAILED"))
    for line in errors:
        print("  - " + line)
    checks = {"no_overlap_and_class_counts": list(errors)}
    if limit is None:
        print("[check] group check and split check: not run (the groups come from a file, so there is no limit)")
    else:
        group = scenes.group_check(unit, groups, limit)
        print(f"[check] group check: {'passed' if not group['failures'] else 'FAILED'}: {group['failures']:,} images have "
              f"an image outside their group at or above the limit {limit:.4f} (the highest such similarity is "
              f"{group['highest_similarity_outside_a_group']:.4f})")
        if group["failures"]:
            errors.append(f"group check: {group['failures']} images have an image outside their group at or above the limit")
        split_errors, numbers = scenes.split_check(unit, splits, groups, limit, min_groups)
        print(f"[check] split check: {'passed' if not split_errors else 'FAILED'}: test {groups_in['test']} groups "
              f"({min_groups[0]} needed), val {groups_in['val']} groups ({min_groups[1]} needed); the highest similarity "
              + ", ".join(f"{pair}: {numbers[pair]['highest_similarity']:.4f}"
                          for pair in ("test to train_pool", "val to train_pool", "test to val")))
        for line in split_errors:
            print("  - " + line)
        errors += [f"split check: {line}" for line in split_errors]
        checks.update(group_check=group, split_check={"errors": split_errors, **numbers})
    checks["passed"] = not errors

    if args.sheets_dir:
        for prefix, split in (("scene_split", splits), ("random_split", scenes.random_split(
                n, len(splits["test"]), len(splits["val"]), args.seed))):
            pairs = scenes.scene_check_pairs(unit, split, args.pairs, args.seed)
            sheets = pair_sheets([(ids[a], ids[b], s) for a, b, s in pairs], image_dir, args.sheets_dir, prefix)
            values = [s for _, _, s in pairs]
            print(f"[measure] scene measurement, {prefix}: {len(pairs)} pairs on {len(sheets)} sheets ({args.sheets_dir}/"
                  f"{prefix}_*.jpg), mean similarity {np.mean(values):.3f}, highest {np.max(values):.4f}. Count the "
                  "pairs that show the same camera view.")
        print("[measure] record the two counts. They go in the report. The count is a measurement, not a pass rule "
              "(design §4.5): a scene group is not a camera."
              + (f" Each pair of the scene split is below the limit {limit:.4f}, so its count tells how many test images "
                 "have their camera in the train pool at a lower similarity." if limit is not None else ""))
        members = np.flatnonzero(groups == int(sizes.argmax()))
        picks = np.sort(np.random.default_rng(args.seed).choice(members, size=min(50, len(members)), replace=False))
        sheet = group_sheet(ids[picks].tolist(), image_dir, args.sheets_dir / "largest_group.jpg")
        print(f"[measure] the largest group ({largest:.1%} of the images): {len(picks)} of its images are on {sheet}. "
              "Count the cameras that they show, for the report.")

    meta.update(groups=int(len(sizes)), largest_group_share=largest, groups_per_split=groups_in,
                images={name: int(len(positions)) for name, positions in splits.items()}, boxes_per_class=counts,
                checks=checks)
    scenes.write_split_files(args.splits_dir, miotcd_ids, groups, splits, meta)
    print(f"[scenes] wrote the split files to {args.splits_dir}: " + json.dumps(meta["images"]))
    raise SystemExit(CHECK_FAILED if errors else 0)


if __name__ == "__main__":
    main()
