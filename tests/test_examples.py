"""Canonical graphical-model examples with known answers.

Three classics, each verified three ways: against an independent brute-force
enumeration written right here (sharing the same CPT constants as the model,
so a transcription error cannot hide), against published literature values,
and across engine paths (hard ``given`` vs point-mass ``believe``, floats vs
exact ``Fraction`` arithmetic).

- Sprinkler: Pearl's rain/sprinkler/wet-grass net, numbers from the
  Wikipedia "Bayesian network" article (P(rain | wet grass) = 35.77%).
- Monty Hall: parametrized over the number of doors; switching wins with
  probability (n-1) / (n·(n-2)), the classic 2/3 at n=3.
- Burglary alarm: Russell & Norvig, "Artificial Intelligence: A Modern
  Approach" — P(Burglary | JohnCalls, MaryCalls) ≈ 0.284.
"""

import pytest
from fractions import Fraction
from itertools import product

from dicekit import Dice, model, p


# --- sprinkler --------------------------------------------------------------

RAIN_P = 0.2
SPRINKLER_P = {True: 0.01, False: 0.4}  # P(sprinkler on | rain)
WET_P = {  # P(wet grass | sprinkler, rain)
    (False, False): 0.0,
    (False, True): 0.8,
    (True, False): 0.9,
    (True, True): 0.99,
}


def _sprinkler_model():
    with model() as m:
        rain = Dice({True: RAIN_P, False: 1 - RAIN_P})
        sprinkler = Dice.from_function(
            lambda r: {True: SPRINKLER_P[r], False: 1 - SPRINKLER_P[r]},
            parents=[rain],
        )
        wet = Dice.from_function(
            lambda s, r: {True: WET_P[(s, r)], False: 1 - WET_P[(s, r)]},
            parents=[sprinkler, rain],
        )
    return m, rain, sprinkler, wet


def _sprinkler_joint():
    for rain, sprinkler, wet in product([True, False], repeat=3):
        weight = RAIN_P if rain else 1 - RAIN_P
        weight *= SPRINKLER_P[rain] if sprinkler else 1 - SPRINKLER_P[rain]
        weight *= WET_P[(sprinkler, rain)] if wet else 1 - WET_P[(sprinkler, rain)]
        yield (rain, sprinkler, wet), weight


def _sprinkler_brute(event, given=None):
    numerator = sum(w for a, w in _sprinkler_joint() if (given is None or given(*a)) and event(*a))
    denominator = sum(w for a, w in _sprinkler_joint() if given is None or given(*a))
    return numerator / denominator


@pytest.mark.parametrize(
    "make_given, brute_given, literature",
    [
        (lambda r, s, w: w == True, lambda rv, sv, wv: wv, 0.3577),
        (lambda r, s, w: (w == True) & (s == True), lambda rv, sv, wv: wv and sv, None),
        # wet grass with the sprinkler off leaves only one explanation
        (lambda r, s, w: (w == True) & (s == False), lambda rv, sv, wv: wv and not sv, 1.0),
        (lambda r, s, w: w == False, lambda rv, sv, wv: not wv, None),
        (lambda r, s, w: s == True, lambda rv, sv, wv: sv, None),
    ],
    ids=["wet", "wet+sprinkler", "wet+no-sprinkler", "dry", "sprinkler-only"],
)
def test_sprinkler_rain_posterior(make_given, brute_given, literature):
    m, rain, sprinkler, wet = _sprinkler_model()
    got = p(rain == True, given=make_given(rain, sprinkler, wet))
    assert got == pytest.approx(_sprinkler_brute(lambda rv, sv, wv: rv, brute_given), abs=1e-12)
    if literature is not None:
        assert got == pytest.approx(literature, abs=5e-4)


def test_sprinkler_wikipedia_numbers():
    m, rain, sprinkler, wet = _sprinkler_model()
    # the two headline numbers from the Wikipedia article
    assert p(rain == True, given=wet == True) == pytest.approx(0.35768, abs=1e-5)
    assert p(sprinkler == True, given=wet == True) == pytest.approx(0.64672, abs=1e-5)
    # explaining away: seeing the sprinkler on makes rain *less* likely
    assert p(rain == True, given=(wet == True) & (sprinkler == True)) < p(rain == True, given=wet == True)


def test_sprinkler_exact_fractions():
    # Fractions in → Fractions out: the engine never leaves exact arithmetic
    fifth = Fraction(1, 5)
    with model():
        rain = Dice({True: fifth, False: 1 - fifth})
        sprinkler = Dice.from_function(
            lambda r: {True: Fraction(1, 100), False: Fraction(99, 100)}
            if r
            else {True: Fraction(2, 5), False: Fraction(3, 5)},
            parents=[rain],
        )
        wet_cpt = {
            (False, False): Fraction(0),
            (False, True): Fraction(4, 5),
            (True, False): Fraction(9, 10),
            (True, True): Fraction(99, 100),
        }
        wet = Dice.from_function(
            lambda s, r: {True: wet_cpt[(s, r)], False: 1 - wet_cpt[(s, r)]},
            parents=[sprinkler, rain],
        )
    posterior = p(rain == True, given=wet == True)
    assert isinstance(posterior, Fraction)
    assert posterior == Fraction(891, 2491)  # == 0.16038 / 0.44838 exactly


def test_sprinkler_soft_belief_is_jeffrey_mixture():
    # Jeffrey conditioning: P'(rain) = q(wet)·P(rain|wet) + q(¬wet)·P(rain|¬wet)
    m, rain, sprinkler, wet = _sprinkler_model()
    wet.believe({True: 0.7, False: 0.3})
    expected = 0.7 * _sprinkler_brute(lambda r, s, w: r, lambda r, s, w: w) + 0.3 * _sprinkler_brute(
        lambda r, s, w: r, lambda r, s, w: not w
    )
    assert p(rain).probs[True] == pytest.approx(expected, abs=1e-9)


# --- monty hall -------------------------------------------------------------


def _monty_model(doors):
    all_doors = tuple(range(1, doors + 1))
    with model() as m:
        car = Dice.from_numbers(*all_doors)
        pick = Dice.from_numbers(*all_doors)
        opened = Dice.from_function(
            lambda c, g: Dice.from_numbers(*[d for d in all_doors if d not in (c, g)]),
            parents=[car, pick],
        )
        final = Dice.from_function(  # switch to a uniformly random other door
            lambda g, o: Dice.from_numbers(*[d for d in all_doors if d not in (g, o)]),
            parents=[pick, opened],
        )
        stay_wins = car == pick
        switch_wins = car == final
    return m, car, pick, opened, stay_wins, switch_wins


@pytest.mark.parametrize("doors", [3, 4, 5])
def test_monty_hall_switching_odds(doors):
    m, car, pick, opened, stay_wins, switch_wins = _monty_model(doors)
    assert p(stay_wins) == pytest.approx(1 / doors, rel=1e-12)
    # host opens one door, player switches to a random remaining one
    assert p(switch_wins) == pytest.approx((doors - 1) / (doors * (doors - 2)), rel=1e-12)


def test_monty_hall_door_posterior_and_belief_equivalence():
    m, car, pick, opened, stay_wins, switch_wins = _monty_model(3)
    # the classic table: you picked door 1, the host opened door 3
    posterior = m.query(car, given=(pick == 1) & (opened == 3))
    assert posterior.probs[1] == pytest.approx(1 / 3, rel=1e-12)
    assert posterior.probs[2] == pytest.approx(2 / 3, rel=1e-12)
    assert posterior.probs.get(3, 0) == 0

    # point-mass beliefs are hard evidence: same numbers, stateful path
    pick.believe({1: 1})
    opened.believe({3: 1})
    assert p(car).probs[2] == pytest.approx(2 / 3, abs=1e-9)
    assert p(switch_wins) == pytest.approx(2 / 3, abs=1e-9)
    m.forget()
    assert p(car).probs[2] == pytest.approx(1 / 3, rel=1e-12)


# --- burglary alarm (Russell & Norvig) ---------------------------------------

BURGLARY_P = 0.001
EARTHQUAKE_P = 0.002
ALARM_P = {  # P(alarm | burglary, earthquake)
    (True, True): 0.95,
    (True, False): 0.94,
    (False, True): 0.29,
    (False, False): 0.001,
}
JOHN_P = {True: 0.90, False: 0.05}  # P(john calls | alarm)
MARY_P = {True: 0.70, False: 0.01}  # P(mary calls | alarm)


def _alarm_model():
    with model() as world:
        burglary = Dice({True: BURGLARY_P, False: 1 - BURGLARY_P})
        earthquake = Dice({True: EARTHQUAKE_P, False: 1 - EARTHQUAKE_P})
        alarm = Dice.from_function(
            lambda b, e: {True: ALARM_P[(b, e)], False: 1 - ALARM_P[(b, e)]},
            parents=[burglary, earthquake],
        )
        john = Dice.from_function(lambda a: {True: JOHN_P[a], False: 1 - JOHN_P[a]}, parents=[alarm])
        mary = Dice.from_function(lambda a: {True: MARY_P[a], False: 1 - MARY_P[a]}, parents=[alarm])
    return world, burglary, earthquake, alarm, john, mary


def _alarm_joint():
    for b, e, a, j, mc in product([True, False], repeat=5):
        weight = BURGLARY_P if b else 1 - BURGLARY_P
        weight *= EARTHQUAKE_P if e else 1 - EARTHQUAKE_P
        weight *= ALARM_P[(b, e)] if a else 1 - ALARM_P[(b, e)]
        weight *= JOHN_P[a] if j else 1 - JOHN_P[a]
        weight *= MARY_P[a] if mc else 1 - MARY_P[a]
        yield (b, e, a, j, mc), weight


def _alarm_brute(event, given=None):
    numerator = sum(w for a, w in _alarm_joint() if (given is None or given(*a)) and event(*a))
    denominator = sum(w for a, w in _alarm_joint() if given is None or given(*a))
    return numerator / denominator


@pytest.mark.parametrize(
    "make_given, brute_given, literature",
    [
        (lambda j, mc: (j == True) & (mc == True), lambda b, e, a, jv, mv: jv and mv, 0.284),
        (lambda j, mc: j == True, lambda b, e, a, jv, mv: jv, None),
        (lambda j, mc: mc == True, lambda b, e, a, jv, mv: mv, None),
        (lambda j, mc: (j == True) & (mc == False), lambda b, e, a, jv, mv: jv and not mv, None),
    ],
    ids=["john+mary", "john", "mary", "john-only-no-mary"],
)
def test_alarm_burglary_posterior(make_given, brute_given, literature):
    world, burglary, earthquake, alarm, john, mary = _alarm_model()
    got = p(burglary == True, given=make_given(john, mary))
    assert got == pytest.approx(_alarm_brute(lambda b, e, a, j, mc: b, brute_given), abs=1e-12)
    if literature is not None:
        assert got == pytest.approx(literature, abs=1e-3)


def test_alarm_all_marginals_match_brute_force():
    world, *variables = _alarm_model()
    john, mary = variables[-2], variables[-1]
    john.believe({True: 1})
    mary.believe({True: 1})
    for index, variable in enumerate(variables):
        expected = _alarm_brute(lambda *a, i=index: a[i], lambda b, e, a, j, mc: j and mc)
        assert p(variable).probs[True] == pytest.approx(expected, abs=1e-9)


def test_alarm_explaining_away():
    world, burglary, earthquake, alarm, john, mary = _alarm_model()
    both_call = (john == True) & (mary == True)
    baseline = p(burglary == True, given=both_call)
    # learning there WAS an earthquake explains the calls away from burglary
    with_quake = p(burglary == True, given=both_call & (earthquake == True))
    assert with_quake < baseline
    assert with_quake == pytest.approx(
        _alarm_brute(lambda b, e, a, j, mc: b, lambda b, e, a, j, mc: j and mc and e), abs=1e-12
    )


def test_alarm_beliefs_match_hard_given():
    world, burglary, earthquake, alarm, john, mary = _alarm_model()
    hard = p(burglary == True, given=(john == True) & (mary == True))
    john.believe({True: 1})
    mary.believe({True: 1})
    assert p(burglary).probs[True] == pytest.approx(hard, abs=1e-9)
