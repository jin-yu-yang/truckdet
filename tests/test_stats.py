import numpy as np
import pytest

from truckdet import stats


def test_resample_weights():
    w = stats.resample_weights(50, n_boot=20, seed=0)
    assert w.shape == (20, 50) and (w.sum(axis=1) == 50).all() and w.min() >= 0
    assert np.array_equal(w, stats.resample_weights(50, 20, 0))  # the same resamples for every run: paired
    assert not np.array_equal(w, stats.resample_weights(50, 20, 1))
    assert np.array_equal(w[:5], stats.resample_weights(50, 5, 0))  # resample i does not depend on n_boot
    assert 0.3 < (w == 0).mean() < 0.45  # about 1/e of the images are not in a resample
    assert stats.N_BOOT == 400


def test_group_resample_weights():
    """Design §7.3: a resample draws scene groups, and the weight of an image is the number of draws of its group."""
    groups = np.array([7, 7, 7, 2, 2, 9, 4, 4, 4, 4])  # 4 groups; the numbers of the groups have no meaning
    w = stats.group_resample_weights(groups, n_boot=200, seed=0)
    assert w.shape == (200, 10) and w.min() >= 0
    for members in ([0, 1, 2], [3, 4], [5], [6, 7, 8, 9]):
        assert (w[:, members] == w[:, members[:1]]).all()  # the images of a group have one weight
    per_group = w[:, [0, 3, 5, 6]]
    assert (per_group.sum(axis=1) == 4).all()  # 4 draws of the 4 groups
    assert not (w.sum(axis=1) == 10).all()  # so the number of images changes from resample to resample
    assert np.array_equal(w, stats.group_resample_weights(groups, 200, 0))
    assert np.array_equal(w[:5], stats.group_resample_weights(groups, 5, 0))  # resample i does not depend on n_boot
    assert not np.array_equal(w, stats.group_resample_weights(groups, 200, 1))
    relabelled = np.array([1, 1, 1, 0, 0, 3, 2, 2, 2, 2])  # other numbers in another order: the same partition
    assert (stats.group_resample_weights(relabelled, 50, 0)[:, [0, 3, 5, 6]].sum(axis=1) == 4).all()
    # one image in each group: the bootstrap of single images
    assert np.array_equal(stats.group_resample_weights(np.arange(50), 20, 0), stats.resample_weights(50, 20, 0))


def test_group_bootstrap_is_wider_than_the_image_bootstrap_when_scenes_differ():
    """Frames of one camera are not independent: with a score that depends on the scene, the image bootstrap
    gives a standard error that is too small."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    score = rng.normal(size=20)[groups] + rng.normal(size=1000) * 0.1  # the scene decides the score of a frame
    def se(weights):
        return float(((weights * score).sum(axis=1) / weights.sum(axis=1)).std(ddof=1))
    by_group, by_image = se(stats.group_resample_weights(groups, 400)), se(stats.resample_weights(1000, 400))
    assert by_group > 4 * by_image


def test_interval_and_se():
    samples = np.arange(101, dtype=float)
    assert stats.interval(samples) == pytest.approx((2.5, 97.5))
    assert stats.interval(np.array([1.0, np.nan, 3.0])) == pytest.approx((1.05, 2.95))
    assert stats.bootstrap_se(np.array([1.0, 2.0, 3.0, np.nan])) == pytest.approx(1.0)
    assert np.isnan(stats.bootstrap_se(np.array([1.0])))


def test_seed_pairings():
    assert stats.seed_pairings([0.5, 0.6, 0.7], [0.4, 0.55, 0.65]) == (6, 3, 9)
    assert stats.seed_pairings([0.4], [0.4 + 1e-12]) == (0, 0, 1)  # float noise is a tie
    assert stats.pairs_in_agreement([0.5, 0.6, 0.7], [0.4, 0.55, 0.65]) == (6, 9)
    assert stats.pairs_in_agreement([0.4, 0.55, 0.65], [0.5, 0.6, 0.7]) == (6, 9)  # the mean difference is negative
    assert stats.pairs_in_agreement([0.5, 0.5], [0.5, 0.5]) == (0, 4)


def test_recovered_share():
    assert float(stats.recovered_share(0.50, 0.40, 0.60)) == pytest.approx(0.5)
    assert float(stats.recovered_share(0.40, 0.40, 0.60)) == 0.0 and float(stats.recovered_share(0.60, 0.40, 0.60)) == 1.0
    assert float(stats.recovered_share(0.65, 0.40, 0.60)) == pytest.approx(1.25)  # an arm above F is above 1
    shares = stats.recovered_share(np.array([0.5, 0.5]), np.array([0.4, 0.5]), np.array([0.6, 0.5]))
    assert shares[0] == pytest.approx(0.5) and np.isnan(shares[1])  # F equal to H: no gain to recover


def test_readings_use_the_words_of_the_design():
    assert (stats.CONFIRMED, stats.REFUTED, stats.NOT_SHOWN) == ("confirmed", "refuted", "not shown")
    # P1: the interval of AP_B - AP_D above zero, below zero, or neither
    assert stats.reading_p1((0.004, 0.031)) == "confirmed"
    assert stats.reading_p1((-0.031, -0.004)) == "refuted"
    assert stats.reading_p1((-0.004, 0.031)) == "not shown"
    assert stats.reading_p1((0.0, 0.031)) == "not shown"  # "above zero" is strict
    assert stats.reading_p1((np.nan, np.nan)) == "not shown"
    # P3: the interval of R_B against 2/3
    assert stats.reading_p3((0.70, 0.95)) == "confirmed"
    assert stats.reading_p3((0.30, 0.60)) == "refuted"
    assert stats.reading_p3((0.60, 0.80)) == "not shown"
    assert stats.P3_THRESHOLD == pytest.approx(2 / 3)


def test_reading_p2_needs_labels_above_zero_and_boxes_at_zero():
    up, down, around = (0.01, 0.05), (-0.05, -0.01), (-0.01, 0.02)
    assert stats.reading_p2(label_b_minus_d=up, localized_b_minus_d=around) == "confirmed"
    assert stats.reading_p2(up, up) == "not shown"  # the boxes are better too: the advantage is not only in the labels
    assert stats.reading_p2(up, down) == "not shown"
    assert stats.reading_p2(around, around) == "not shown"
    assert stats.reading_p2(down, around) == "refuted" and stats.reading_p2(down, up) == "refuted"
    assert stats.reading_p2(up, (np.nan, np.nan)) == "not shown"


def test_gate():
    rng = np.random.default_rng(0)
    h = 0.40 + rng.normal(0, 0.01, 400)
    passed = stats.gate(0.40, 0.52, h, h + 0.12 + rng.normal(0, 0.005, 400))
    assert passed["passed"] and passed["difference"] == pytest.approx(0.12) and passed["ratio"] > 5
    assert passed["bootstrap_se"] == pytest.approx(0.005, rel=0.2)  # the se of the difference, on paired resamples
    small = stats.gate(0.40, 0.44, h, h + 0.04 + rng.normal(0, 0.001, 400))
    assert not small["passed"] and small["ratio"] > 5  # sure, but less than 0.05 AP
    noisy = stats.gate(0.40, 0.46, h, h + 0.06 + rng.normal(0, 0.03, 400))
    assert not noisy["passed"] and noisy["difference"] >= 0.05  # 0.05 AP, but less than 5 standard errors
    assert not stats.gate(0.50, 0.40, h, h - 0.1)["passed"]


def test_choose_lr():
    assert stats.choose_lr({1e-4: 0.412, 3e-4: 0.431, 1e-3: 0.398}) == 3e-4
    assert stats.choose_lr({1e-4: 0.4312, 3e-4: 0.4308, 1e-3: 0.398}) == 1e-4  # equal to 3 decimal places: the lower rate
    assert stats.choose_lr({1e-4: 0.4304, 3e-4: 0.4316}) == 3e-4  # 0.430 against 0.432
    with pytest.raises(ValueError):
        stats.choose_lr({})


def test_lr_to_add_is_one_step_one_time():
    grid = [1e-4, 3e-4, 1e-3]
    assert stats.lr_to_add(grid, {1e-4: 0.41, 3e-4: 0.43, 1e-3: 0.40}) is None  # the best rate is inside the list
    assert stats.lr_to_add(grid, {1e-4: 0.45, 3e-4: 0.43, 1e-3: 0.40}) == 3e-5
    assert stats.lr_to_add(grid, {1e-4: 0.41, 3e-4: 0.43, 1e-3: 0.47}) == 3e-3
    assert stats.lr_to_add(grid, {3e-5: 0.46, 1e-4: 0.45, 3e-4: 0.43, 1e-3: 0.40}) is None  # one time only
    assert stats.lr_to_add(grid, {1e-4: 0.45, 3e-4: 0.43}) is None  # the list is not complete
    assert stats.lr_to_add([1e-5, 3e-5, 1e-4], {1e-5: 0.5, 3e-5: 0.4, 1e-4: 0.3}) == 3e-6
