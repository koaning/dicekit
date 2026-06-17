import pytest
from fractions import Fraction
from itertools import product
from statistics import NormalDist
from dicekit import Dice, Vase, p, exp, mix, var, ordered


def brute_force_ordered(*dice_in):
    dice_out = [{} for _ in dice_in]
    for combo in product(*[die.probs.items() for die in dice_in]):
        outcomes = sorted([outcome for outcome, _ in combo], reverse=True)
        probability = 1
        for _, outcome_probability in combo:
            probability *= outcome_probability
        for index, outcome in enumerate(outcomes):
            dice_out[index][outcome] = dice_out[index].get(outcome, 0) + probability
    return [Dice(probs) for probs in dice_out]


def old_method_ordered(*dice_in):
    result = {}
    for combo in product(*[die.probs.items() for die in dice_in]):
        eyes = tuple(sorted([outcome for outcome, _ in combo], reverse=True))
        probability = 1
        for _, outcome_probability in combo:
            probability *= outcome_probability
        result[eyes] = result.get(eyes, 0) + probability

    dice_out = []
    for index in range(len(dice_in)):
        new_dice = {}
        for keys, probability in result.items():
            new_dice[keys[index]] = new_dice.get(keys[index], 0) + probability
        dice_out.append(Dice(new_dice))
    return dice_out


def assert_dice_probs_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_dice, expected_dice in zip(actual, expected):
        keys = set(actual_dice.probs) | set(expected_dice.probs)
        for key in keys:
            assert actual_dice.probs.get(key, 0) == pytest.approx(
                expected_dice.probs.get(key, 0)
            )


def test_dice_creation():
    # Test basic dice creation
    d6 = Dice.from_sides(6)
    assert len(d6.probs) == 6
    assert all(abs(v - 1/6) < 1e-10 for v in d6.probs.values())

    # Test custom dice creation
    custom = Dice({1: 0.5, 2: 0.5})
    assert len(custom.probs) == 2
    assert custom.probs[1] == 0.5

def test_dice_from_numbers():
    dice = Dice.from_numbers(1, 1, 2)

    assert dice.probs == {1: pytest.approx(2 / 3), 2: pytest.approx(1 / 3)}

def test_dice_operations():
    d1 = Dice({1: 0.5, 2: 0.5})
    d2 = Dice({1: 0.5, 2: 0.5})
    
    # Test addition
    d_sum = d1 + d2
    assert d_sum.probs[2] == 0.25  # P(1+1)
    assert d_sum.probs[3] == 0.5   # P(1+2 or 2+1)
    assert d_sum.probs[4] == 0.25  # P(2+2)

    # Test multiplication
    d_mult = d1 * d2
    assert d_mult.probs[1] == 0.25  # P(1*1)
    assert d_mult.probs[2] == 0.5   # P(1*2 or 2*1)
    assert d_mult.probs[4] == 0.25  # P(2*2)

def test_dice_map():
    dice = Dice({1: 0.25, 2: 0.25, 3: 0.5})

    parity = dice.map(lambda value: "even" if value % 2 == 0 else "odd")
    assert parity.probs == {"odd": 0.75, "even": 0.25}

    labels = dice.map({1: "low", 2: "low", 3: "high"})
    assert labels.probs == {"low": 0.5, "high": 0.5}

def test_dice_mix():
    first = Dice({"a": 1})
    second = Dice({"b": 1})

    even = mix(first, second)
    assert even.probs == {"a": 0.5, "b": 0.5}

    weighted = mix(first, second, weights=[1, 3])
    assert weighted.probs == {"a": 0.25, "b": 0.75}

def test_dice_mix_validates_arguments():
    dice = Dice({1: 1})

    with pytest.raises(TypeError, match="Dice objects"):
        mix(dice, "not a dice")
    with pytest.raises(ValueError, match="one value"):
        mix(dice, dice, weights=[1])
    with pytest.raises(ValueError, match="non-negative"):
        mix(dice, dice, weights=[1, -1])
    with pytest.raises(ValueError, match="positive sum"):
        mix(dice, dice, weights=[0, 0])
    with pytest.raises(ValueError, match="at least one"):
        mix()

def test_dice_sum():
    d6 = Dice.from_sides(6)
    three_d6 = sum([d6, d6, d6])
    assert three_d6.probs[10] == pytest.approx(27 / 216)
    assert three_d6.probs[3] == pytest.approx(1 / 216)
    assert sum([d6]).probs == d6.probs

    labels = Dice({"a": 0.5, "b": 0.5})
    assert sum([labels, labels]).probs == {
        "aa": 0.25,
        "ab": 0.25,
        "ba": 0.25,
        "bb": 0.25,
    }
    assert (labels + 0).probs == labels.probs
    assert (0 + labels).probs == labels.probs

    d_with_zero = Dice({0: 0.5, 1: 0.5})
    assert sum([d_with_zero, d_with_zero]).probs == {0: 0.25, 1: 0.5, 2: 0.25}

def test_dice_reflected_operations():
    d = Dice({1: 0.5, 2: 0.5})

    d_sum = 2 + d
    assert d_sum.probs[3] == 0.5
    assert d_sum.probs[4] == 0.5

    d_sub = 2 - d
    assert d_sub.probs[1] == 0.5
    assert d_sub.probs[0] == 0.5

    d_mult = 2 * d
    assert d_mult.probs[2] == 0.5
    assert d_mult.probs[4] == 0.5

def test_dice_comparisons():
    d1 = Dice({1: 0.5, 2: 0.5})
    d2 = Dice({1: 0.5, 2: 0.5})
    
    # Test greater than
    d_gt = d1 > d2
    assert p(d_gt) == 0.25  # P(2 > 1) = 0.25

    # Test less than or equal
    d_le = d1 <= d2
    assert p(d_le) == 0.75  # P(1 ≤ 1 or 1 ≤ 2 or 2 ≤ 2) = 0.75

def test_dice_equality_with_categorical_value():
    draws = Vase.from_counts(a=3, b=3).take(3, replace=True, ordered=False)

    assert p(draws == "bbb") == pytest.approx(1 / 8)
    assert p(draws != "bbb") == pytest.approx(7 / 8)
    assert p(draws == "ccc") == 0

def test_fraction_probs_stay_exact():
    d = Dice({1: Fraction(1), 2: Fraction(1), 3: Fraction(1)})

    # probabilities remain exact Fractions, not floats
    assert all(isinstance(v, Fraction) for v in d.probs.values())
    assert d.probs == {1: Fraction(1, 3), 2: Fraction(1, 3), 3: Fraction(1, 3)}

    # exp stays exact and arithmetic keeps Fraction values
    assert exp(d) == Fraction(2)
    summed = d + d
    assert all(isinstance(v, Fraction) for v in summed.probs.values())
    assert summed.probs[4] == Fraction(1, 3)


def test_fraction_outcomes():
    d = Dice({Fraction(1, 2): 1, Fraction(3, 2): 1})
    shifted = d + Fraction(1, 2)
    assert set(shifted.probs) == {Fraction(1), Fraction(2)}

    # Fraction outcomes dedupe with equal ints (hash(F(3)) == hash(3))
    mixed = Dice({Fraction(3, 1): 1}) + Dice({0: 1})
    assert list(mixed.probs) == [3]


def test_ordered_preserves_fraction_exactness():
    d3 = Dice({1: Fraction(1, 3), 2: Fraction(1, 3), 3: Fraction(1, 3)})
    highest, lowest = ordered(d3, d3)

    # results stay exact, and agree with the float computation
    assert all(isinstance(v, Fraction) for v in highest.probs.values())
    assert highest.probs[3] == Fraction(5, 9)  # P(max == 3) = 1 - (2/3)^2
    float_highest = Dice.from_sides(3).out_of(2, max)
    for outcome, prob in highest.probs.items():
        assert float(prob) == pytest.approx(float_highest.probs[outcome])


def test_charts_render_with_fractions():
    import marimo as mo

    cases = [
        Dice({1: Fraction(1), 2: Fraction(1)}),        # Fraction probabilities
        Dice({Fraction(1, 2): 1, Fraction(3, 2): 1}),  # Fraction outcomes
    ]
    for d in cases:
        # these used to raise "Object of type Fraction is not JSON serializable"
        assert mo.as_html(d.prob_chart()).text
        assert mo.as_html(d.cdf_chart()).text


def test_charts_still_render_with_string_keys():
    import marimo as mo

    draws = Vase.from_counts(a=2, b=1).take(2, replace=True)
    assert all(isinstance(k, str) for k in draws.probs)
    assert mo.as_html(draws.prob_chart()).text


def test_dice_filter():
    d6 = Dice.from_sides(6)
    d_even = d6.filter(lambda x: x % 2 == 0)
    assert len(d_even.probs) == 3  # Only 2, 4, 6
    assert abs(sum(d_even.probs.values()) - 1.0) < 1e-10

def test_dice_out_of():
    d6 = Dice.from_sides(6)
    max_of_2 = d6.out_of(2, max)
    # P(getting a 6 from 2d6) = 11/36
    assert abs(max_of_2.probs[6] - 11/36) < 1e-10

def test_statistics():
    # Test a fair d6
    d6 = Dice.from_sides(6)
    
    # Expected value should be 3.5
    assert abs(exp(d6) - 3.5) < 1e-10
    
    # Variance should be 35/12 ≈ 2.916667
    assert abs(var(d6) - 35/12) < 1e-10

def test_dice_roll():
    d6 = Dice.from_sides(6)
    rolls = d6.roll(1000)
    assert len(rolls) == 1000
    assert all(1 <= r <= 6 for r in rolls)

def test_dice_ordered_2():
    d6 = Dice.from_sides(6)
    a1, a2 = ordered(d6, d6)
    assert a1.probs[1] == pytest.approx(1/6/6)
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10
    for k, v in a2.probs.items():
        assert abs(v - d6.out_of(2, min).probs[k]) < 1e-10

def test_dice_ordered_method():
    d6 = Dice.from_sides(6)
    a1, a2 = d6.ordered(2)
    assert a1.probs[1] == pytest.approx(1/6/6)
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10
    for k, v in a2.probs.items():
        assert abs(v - d6.out_of(2, min).probs[k]) < 1e-10

    highest, = d6.ordered(2, k=1)
    for k, v in highest.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10

def test_dice_ordered_method_limits_work_to_k_results():
    d6 = Dice.from_sides(6)

    first_two = d6.ordered(5, k=2)
    all_ordered = d6.ordered(5)

    assert len(first_two) == 2
    assert_dice_probs_close(first_two, all_ordered[:2])

def test_dice_ordered_3():
    d6 = Dice.from_sides(6)
    a1, _, a3 = ordered(d6, d6, d6)
    assert a1.probs[1] == pytest.approx(1/6/6/6)
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(3, max).probs[k]) < 1e-8
    for k, v in a3.probs.items():
        assert abs(v - d6.out_of(3, min).probs[k]) < 1e-8

@pytest.mark.parametrize(
    "dice",
    [
        (
            Dice.from_sides(4),
            Dice.from_sides(6),
            Dice({1: 0.1, 5: 0.2, 9: 0.7}),
        ),
        (
            Dice({1: 1, 3: 2, 10: 1}),
            Dice({3: 1, 10: 1}),
            Dice({1: 4, 10: 1}),
        ),
        (
            Dice.from_sides(3),
            Dice({0: 1, 2: 2}),
            Dice({1: 1, 4: 1}),
            Dice({-1: 1, 5: 3}),
        ),
    ],
)
def test_dice_ordered_matches_brute_force(dice):
    assert_dice_probs_close(ordered(*dice), brute_force_ordered(*dice))

def test_dice_ordered_mixed_dice_matches_old_method():
    dice = [
        Dice.from_sides(4),
        Dice.from_sides(6),
        Dice({1: 0.1, 5: 0.2, 9: 0.7}),
        Dice({0: 2, 4: 1}),
    ]

    assert_dice_probs_close(ordered(*dice), old_method_ordered(*dice))

def test_from_dist_equiprobable_faces():
    dice = Dice.from_dist(NormalDist(0, 1), n=10)
    assert len(dice.probs) == 10
    assert all(v == pytest.approx(0.1) for v in dice.probs.values())
    assert sum(dice.probs.values()) == pytest.approx(1.0)


def test_from_dist_symmetric_normal():
    dice = Dice.from_dist(NormalDist(0, 1), n=10)
    outcomes = sorted(dice.probs)
    for low, high in zip(outcomes, reversed(outcomes)):
        assert low == pytest.approx(-high)
    assert exp(dice) == pytest.approx(0.0, abs=1e-9)


def test_from_dist_monotonic_midpoints():
    n = 8
    dice = Dice.from_dist(NormalDist(0, 1), n=n)
    outcomes = [NormalDist(0, 1).inv_cdf((i + 0.5) / n) for i in range(n)]
    assert all(a < b for a, b in zip(outcomes, outcomes[1:]))
    assert set(dice.probs) == set(outcomes)


def test_from_dist_custom_quantiles():
    dist = NormalDist(0, 1)
    dice = Dice.from_dist(dist, quantiles=[0.25, 0.5, 0.75])
    assert len(dice.probs) == 3
    assert all(v == pytest.approx(1 / 3) for v in dice.probs.values())
    expected = {dist.inv_cdf(q) for q in (0.25, 0.5, 0.75)}
    assert set(dice.probs) == expected


def test_from_dist_duck_typed_ppf():
    class StubDist:
        def ppf(self, q):
            return q

    dice = Dice.from_dist(StubDist(), n=4)
    assert len(dice.probs) == 4
    assert set(dice.probs) == {0.125, 0.375, 0.625, 0.875}


def test_from_dist_mean_tracks_distribution():
    dice = Dice.from_dist(NormalDist(5, 2), n=50)
    assert exp(dice) == pytest.approx(5, abs=0.05)


def test_from_dist_requires_inverse_cdf():
    with pytest.raises(TypeError):
        Dice.from_dist(object())


@pytest.mark.parametrize("quantiles", [[], [0.0, 0.5], [0.5, 1.0]])
def test_from_dist_rejects_bad_quantiles(quantiles):
    with pytest.raises(ValueError):
        Dice.from_dist(NormalDist(0, 1), quantiles=quantiles)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dice_ordered_k_matches_prefix(k):
    dice = [
        Dice.from_sides(4),
        Dice.from_sides(6),
        Dice({1: 0.1, 5: 0.2, 9: 0.7}),
    ]

    first_two = ordered(*dice, k=k)
    all_ordered = ordered(*dice)

    assert len(first_two) == k
    assert_dice_probs_close(first_two, all_ordered[:k])
