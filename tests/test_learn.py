from functools import partial

import pytest

from dicekit import Dice, exp, p
from dicekit.learn import DAG, Event, Node


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
