from functools import partial

import pytest

from dicekit import Dice, exp, p
from dicekit.learn import DAG, Event, Node, model


def disease_dag():
    """A small disease/test Bayesian network used across the tests."""
    dag = DAG()
    dag.add_variable("disease", {"present": 0.01, "absent": 0.99})

    def test_result(disease, sensitivity, false_positive_rate):
        if disease == "present":
            return {"positive": sensitivity, "negative": 1 - sensitivity}
        return {"positive": false_positive_rate, "negative": 1 - false_positive_rate}

    dag.add_variable(
        "test_1",
        given="disease",
        dist=partial(test_result, sensitivity=0.90, false_positive_rate=0.05),
    )
    return dag


def test_p_on_node_returns_posterior_dice():
    dag = disease_dag()
    disease, test_1 = dag.get_variables("disease, test_1")

    posterior = p(disease, given=test_1 == "positive")

    assert isinstance(posterior, Dice)
    # Bayes: P(present | +) = .01*.90 / (.01*.90 + .99*.05)
    assert posterior.probs["present"] == pytest.approx(0.1538, abs=1e-4)
    assert posterior.probs["absent"] == pytest.approx(0.8462, abs=1e-4)


def test_p_on_event_returns_scalar():
    dag = disease_dag()
    test_1 = dag.get_variables("test_1")

    # marginal P(test_1 == positive) = .01*.90 + .99*.05
    prob = p(test_1 == "positive")

    assert isinstance(prob, float)
    assert prob == pytest.approx(0.0585, abs=1e-4)


def test_node_equality_builds_an_event():
    dag = disease_dag()
    test_1 = dag.get_variables("test_1")

    assert isinstance(test_1 == "positive", Event)
    assert isinstance(test_1, Node)


def test_core_helpers_consume_dag_output():
    # A Dice produced by the DAG flows into the core helpers unchanged.
    dag = disease_dag()
    disease = dag.get_variables("disease")

    prior = p(disease)
    assert isinstance(prior, Dice)
    # exp over a {present/absent} dice has no numeric meaning, but a numeric
    # marginal does: map outcomes to 0/1 and check exp matches the prior.
    indicator = prior.map({"present": 1, "absent": 0})
    assert exp(indicator) == pytest.approx(0.01)


# --- puzzle API: model / variable ------------------------------------------


def monty_model():
    """The Monty Hall puzzle in the puzzle API."""
    with model():
        car = Dice.from_numbers(1, 2, 3)
        pick = Dice.from_numbers(1, 2, 3)
        opened = Dice.from_function(
            lambda c, g: Dice.from_numbers(*[d for d in (1, 2, 3) if d != c and d != g]),
            parents=[car, pick],
        )
        switch = Dice.from_numbers(True, False)
        win = (switch & (car != pick)) | (~switch & (car == pick))

    return car, pick, opened, switch, win


def test_monty_switching_beats_staying():
    car, pick, opened, switch, win = monty_model()

    assert p(win, given=switch == True) == pytest.approx(2 / 3)
    assert p(win, given=switch == False) == pytest.approx(1 / 3)
    # Observing which door the host opened does not change the switch odds.
    assert p(win, given=(switch == True) & (opened == 3)) == pytest.approx(2 / 3)


def test_predicate_returns_scalar_variable_returns_dice():
    car, pick, opened, switch, win = monty_model()

    assert isinstance(p(win), float)          # win is a predicate -> P(True)
    assert isinstance(p(car), Dice)           # car is a variable  -> marginal Dice
    assert p(car).probs[1] == pytest.approx(1 / 3)


def test_variables_capture_their_assigned_names():
    with model() as m:
        car = Dice.from_numbers(1, 2, 3)
        opened = Dice.from_function(lambda c: Dice.from_numbers(*[d for d in (1, 2, 3) if d != c]), parents=[car])

    assert [source._name for source in m.sources] == ["car", "opened"]


def test_show_graph_includes_named_derived_variable_and_its_edges():
    with model() as m:
        car = Dice.from_numbers(1, 2, 3)
        pick = Dice.from_numbers(1, 2, 3)
        switch = Dice.from_numbers(True, False)
        win = (switch & (car != pick)) | (~switch & (car == pick))

    assert [d._name for d in m.derived] == ["win"]

    widget = m.show()
    assert set(widget.nodes) == {"car", "pick", "switch", "win"}
    # win depends on all three sources; each is drawn as an edge into it.
    assert {tuple(e) for e in widget.edges} == {("car", "win"), ("pick", "win"), ("switch", "win")}


def test_show_inputs_adds_control_node_keeping_arcs():
    import marimo as mo

    slider = mo.ui.slider(0.0, 1.0, value=0.01)
    with model() as m:
        disease = Dice({"present": slider.value, "absent": 1 - slider.value})
        test = Dice.from_function(lambda d: {"pos": 0.9, "neg": 0.1}, parents=[disease])

    widget = m.show(inputs={disease: slider})

    assert "disease input" in widget.nodes
    assert widget.nodes["disease input"] is slider
    edges = {tuple(e) for e in widget.edges}
    assert ("disease", "test") in edges            # original arc kept
    assert ("disease input", "disease") in edges   # control wired in


def test_show_given_charts_posterior_not_prior():
    # With `given`, each node charts its posterior over the conditioned joint.
    with model() as m:
        disease = Dice({"present": 0.01, "absent": 0.99})
        test = Dice.from_function(
            lambda d: {"positive": 0.90, "negative": 0.10}
            if d == "present"
            else {"positive": 0.05, "negative": 0.95},
            parents=[disease],
        )

    prior = m.query(disease)
    posterior = m.query(disease, given=test == "positive")
    assert prior.probs["present"] == pytest.approx(0.01)
    assert posterior.probs["present"] == pytest.approx(0.1538, abs=1e-4)

    # show(given=...) renders the posterior; edges are identical to show().
    plain = m.show()
    conditioned = m.show(given=test == "positive")
    assert {tuple(e) for e in plain.edges} == {tuple(e) for e in conditioned.edges}
    assert set(conditioned.nodes) == {"disease", "test"}


def test_disease_puzzle_matches_dag():
    with model():
        disease = Dice({"present": 0.01, "absent": 0.99})
        test = Dice.from_function(
            lambda d: {"positive": 0.90, "negative": 0.10}
            if d == "present"
            else {"positive": 0.05, "negative": 0.95},
            parents=[disease],
        )

    assert p(disease == "present", given=test == "positive") == pytest.approx(0.1538, abs=1e-4)
    posterior = p(disease, given=test == "positive")
    assert isinstance(posterior, Dice)
    assert posterior.probs["absent"] == pytest.approx(0.8462, abs=1e-4)
    assert p(test == "positive") == pytest.approx(0.0585, abs=1e-4)


def test_zero_probability_evidence_raises():
    with model():
        a = Dice.from_numbers(1, 2)
        # b is fully determined by a: "hit" iff a == 1.
        b = Dice.from_function(lambda x: {"hit" if x == 1 else "miss": 1.0}, parents=[a])

    # (a == 2) & (b == "hit") can never happen, so conditioning on it raises.
    with pytest.raises(ValueError):
        p(a, given=(a == 2) & (b == "hit"))
