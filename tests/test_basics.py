import pytest
from dicekit import Dice, Vase, p, exp, mix, var, ordered

def test_dice_creation():
    # Test basic dice creation
    d6 = Dice.from_sides(6)
    assert len(d6.probs) == 6
    assert all(abs(v - 1/6) < 1e-10 for v in d6.probs.values())

    # Test custom dice creation
    custom = Dice({1: 0.5, 2: 0.5})
    assert len(custom.probs) == 2
    assert custom.probs[1] == 0.5

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

def test_dice_reflected_operations():
    d = Dice({1: 0.5, 2: 0.5})

    d_sum = 2 + d
    assert d_sum.probs[3] == 0.5
    assert d_sum.probs[4] == 0.5

    d_sub = 2 - d
    assert d_sub.probs[1] == 0.5
    assert d_sub.probs[0] == 0.5

    d_mult = 2 @ d
    assert d_mult.probs[2] == 0.5
    assert d_mult.probs[4] == 0.5

def test_dice_mul_counts_rolls():
    d6 = Dice.from_sides(6)

    three_d6 = d6 * 3
    assert abs(exp(three_d6) - exp(d6 + d6 + d6)) < 1e-10
    assert abs(three_d6.probs[3] - (1 / 6) ** 3) < 1e-10

    reflected = 3 * d6
    assert abs(exp(reflected) - exp(three_d6)) < 1e-10

def test_dice_matmul_scales_values():
    d = Dice({1: 0.5, 2: 0.5})

    scaled = d @ 2
    assert scaled.probs[2] == 0.5
    assert scaled.probs[4] == 0.5

    reflected = 2 @ d
    assert reflected.probs == scaled.probs

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
    assert a1.probs[1] == 1/6/6
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10
    for k, v in a2.probs.items():
        assert abs(v - d6.out_of(2, min).probs[k]) < 1e-10

def test_dice_ordered_method():
    d6 = Dice.from_sides(6)
    a1, a2 = d6.ordered(2)
    assert a1.probs[1] == 1/6/6
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10
    for k, v in a2.probs.items():
        assert abs(v - d6.out_of(2, min).probs[k]) < 1e-10

    highest, = d6.ordered(2, k=1)
    for k, v in highest.probs.items():
        assert abs(v - d6.out_of(2, max).probs[k]) < 1e-10

def test_dice_ordered_3():
    d6 = Dice.from_sides(6)
    a1, _, a3 = ordered(d6, d6, d6)
    assert a1.probs[1] == pytest.approx(1/6/6/6)
    for k, v in a1.probs.items():
        assert abs(v - d6.out_of(3, max).probs[k]) < 1e-8
    for k, v in a3.probs.items():
        assert abs(v - d6.out_of(3, min).probs[k]) < 1e-8
