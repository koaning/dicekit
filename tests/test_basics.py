from dicekit import Dice, p, exp, var

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

def test_dice_comparisons():
    d1 = Dice({1: 0.5, 2: 0.5})
    d2 = Dice({1: 0.5, 2: 0.5})
    
    # Test greater than
    d_gt = d1 > d2
    assert p(d_gt) == 0.25  # P(2 > 1) = 0.25

    # Test less than or equal
    d_le = d1 <= d2
    assert p(d_le) == 0.75  # P(1 ≤ 1 or 1 ≤ 2 or 2 ≤ 2) = 0.75

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
