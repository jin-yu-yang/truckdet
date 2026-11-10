"""A test of the scene-group procedure of design §4.3 on invented cameras.

This is NOT evidence about MIO-TCD. The camera model below is invented. The test shows that the
mechanics of the procedure work, and that its failure signals work. It cannot show that the cameras
of MIO-TCD are separable in the real embedding: only the scene check on the real images shows that.

Camera model: each camera has a centre. A frame of a camera is the centre, plus a time-of-day value
t in [0, 1] times the camera's light vector, plus noise. A part of the light vector is common to all
cameras (dark frames look alike). All frames have a large common part, as the real embeddings have.

The procedure is the code of the project (`truckdet.data.scenes`): the neighbour lists, the groups
for a limit, the candidate limits, the search, the split with a cap for each group, the group check
and the split check. The true camera of each frame takes the place of the person in the scene check.
Two things are here for comparison only: k-means, the first method, which failed the scene check on
MIO-TCD; and a rule that sets the limit from a count by a person, which the design does not use.

    python scripts/scene_split_simulation.py        # four cases, about 2 minutes on a CPU

The k-means line needs scikit-learn (`pip install -e ".[dev]"`); without it the line is left out.
Design §4.3 has the table of the results.
"""

import math

import numpy as np

from truckdet.data import scenes

CASES = {
    "A: cameras well apart": dict(cam_var=0.07, light_var=0.02, shared=0.5),
    "B: cameras nearer, dark frames alike": dict(cam_var=0.03, light_var=0.04, shared=0.8),
    "C: hard": dict(cam_var=0.008, light_var=0.04, shared=0.9),
    "D: not separable": dict(cam_var=0.003, light_var=0.04, shared=0.95, frame_var=0.006),
}


def make_cameras(seed=0, n_cam=150, dim=64, cam_var=0.07, light_var=0.02, shared=0.5, frame_var=0.004, median=100):
    """(unit-length frames [N, dim] float32, the camera of each frame [N])."""
    rng = np.random.default_rng(seed)
    sizes = np.clip(rng.lognormal(np.log(median), 1.0, n_cam).astype(int), 8, 4000)
    common = np.zeros(dim)
    common[0] = 1.0
    shared_dir = rng.normal(size=dim)
    shared_dir[0] = 0
    shared_dir /= np.linalg.norm(shared_dir)
    rows, camera = [], []
    for c, n in enumerate(sizes):
        centre = rng.normal(size=dim)
        centre[0] = 0
        centre *= np.sqrt(cam_var) / np.linalg.norm(centre)
        own = rng.normal(size=dim)
        own[0] = 0
        own /= np.linalg.norm(own)
        light = np.sqrt(light_var) * (np.sqrt(shared) * shared_dir + np.sqrt(1 - shared) * own)
        t = rng.beta(0.6, 0.6, n)  # mostly day or night, some dusk
        noise = rng.normal(size=(n, dim)) * np.sqrt(frame_var / dim)
        noise[:, 0] = 0
        rows.append(common + centre + t[:, None] * light + noise)
        camera += [c] * n
    x = np.concatenate(rows)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), np.array(camera)


def same_camera_share(unit, camera, split, n=200, seed=0) -> float:
    """The scene check, with the true cameras in the place of the person."""
    pairs = scenes.scene_check_pairs(unit, split, n, seed)
    return float(np.mean([camera[a] == camera[b] for a, b, _ in pairs]))


def kmeans_scene_check(x, unit, camera, n_test, n_val) -> float | None:
    """The first method: k-means with one group for each 110 images, whole groups, no cap."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return None
    groups = KMeans(n_clusters=max(round(len(x) / 110), 2), n_init=1, random_state=0).fit_predict(x)
    return same_camera_share(unit, camera, scenes.split_with_caps(groups, n_test, n_val, caps=(None, None)))


def person_rule(unit, camera, limits, seed=1, anchors=2000, per_bin=25) -> float:
    """A rule that the design does not use: a person labels 25 pairs in each bin of similarity, and the limit
    is the similarity below which 1 pair of 25 or fewer shows the same camera."""
    rng = np.random.default_rng(seed)
    x = unit.numpy()
    a = rng.choice(len(x), min(anchors, len(x)), replace=False)
    sims = x[a] @ x.T
    sims[np.arange(len(a)), a] = -1
    edges = limits[::2]
    limit = edges[0]
    for b in range(len(edges) - 2, -1, -1):  # from the bin of lowest similarity upward
        r, c = np.nonzero((sims <= edges[b]) & (sims > edges[b + 1]))
        if len(r) == 0:
            continue
        pick = rng.choice(len(r), min(per_bin, len(r)), replace=False)
        if (camera[a[r[pick]]] == camera[c[pick]]).sum() >= 2:
            limit = edges[b + 1]
            break
    return float(limit)


def run(label: str, n_train: int = 4000, quiet: bool = False, **camera_model) -> dict:
    """One case. Returns the numbers of its row in the table of design §4.3."""
    say = (lambda *a, **k: None) if quiet else print
    x, camera = make_cameras(**camera_model)
    n, n_cameras = len(x), int(camera.max()) + 1
    n_test, n_val = round(n * 10 / 110), round(n * 2 / 110)
    min_groups = (math.ceil(n_test / scenes.TEST_CAP), math.ceil(n_val / scenes.VAL_CAP))
    unit = scenes.as_tensor(x)
    lists = scenes.neighbour_lists(unit)
    d_near, d_rand = scenes.link_distances(unit, lists)
    limits = scenes.candidate_limits(d_near, d_rand)
    say(f"\n=== {label}: {n:,} images, {n_cameras} cameras; nearest-image similarity {1 - d_near:.4f}, "
        f"random-pair similarity {1 - d_rand:.4f}")
    kmeans = kmeans_scene_check(x, unit, camera, n_test, n_val)
    if kmeans is not None:
        say(f"k-means, {max(round(n / 110), 2)} groups: scene check {kmeans:.0%} same camera")

    def try_split(groups):
        split = scenes.split_with_caps(groups, n_test, n_val)
        return split, scenes.split_problems(split, n_test, n_val, n_train)

    def log(row):
        groups, added = scenes.link_groups(unit, lists, row["limit"])
        cut = np.mean([len(set(groups[camera == c].tolist())) > 1 for c in range(n_cameras)])
        line = (f"limit {row['limit']:.4f}: {row['groups']:5d} groups, largest {row['largest_group_share']:5.1%}, cameras "
                f"in 2 or more groups {cut:4.0%}, links that the neighbour lists did not hold {len(added)}")
        if row["over_linked"]:
            line += f"   OVER-LINKED: {row['over_linked']}"
        else:
            line += f"; scene check {same_camera_share(unit, camera, try_split(groups)[0]):4.0%} same camera"
        say(line, flush=True)

    result = scenes.search_limit(unit, lists, limits, try_split, log=None if quiet else log)
    out = {"images": n, "cameras": n_cameras, "kmeans_scene_check": kmeans, "limit": None, "groups": None,
           "scene_check": None, "group_check_failures": None, "split_check_errors": None,
           "candidates": result["candidates"]}
    chosen = result["chosen"]
    if chosen is None:
        say("chosen limit: none. The first limit is over-linked: this embedding does not separate the cameras.")
    else:
        limit, groups, split = chosen["limit"], chosen["groups"], chosen["split"]
        check = scenes.group_check(unit, groups, limit)
        errors, numbers = scenes.split_check(unit, split, groups, limit, min_groups)
        share = same_camera_share(unit, camera, split)
        out.update(limit=limit, groups=int(groups.max()) + 1, scene_check=share, group_check_failures=check["failures"],
                   split_check_errors=errors)
        say(f"chosen limit: {limit:.4f}; {out['groups']} groups; test {len(split['test'])} images in "
            f"{numbers['test groups']} groups, val {len(split['val'])} in {numbers['val groups']}, not used "
            f"{len(split['unused'])}; group check: {check['failures']} failures; split check: {len(errors)} errors; "
            f"scene check {share:.0%} same camera -> " + ("passed" if share <= 0.05 else "FAILED"))
    rule = person_rule(unit, camera, limits)
    groups = scenes.link_groups(unit, lists, rule)[0]
    out["person_rule"] = {"limit": rule, "groups": int(groups.max()) + 1,
                          "largest_group_share": float(np.bincount(groups).max() / n)}
    say(f"for comparison, the limit from a count by a person: {rule:.4f}: {out['person_rule']['groups']} groups, "
        f"largest {out['person_rule']['largest_group_share']:.1%}")
    return out


if __name__ == "__main__":
    for name, model in CASES.items():
        run(name, **model)
