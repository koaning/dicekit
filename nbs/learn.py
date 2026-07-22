# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DAG prototype

    This notebook is a scratchpad for exploring a DAG-flavored API on top
    of the exported `dicekit` package. Nothing in here is exported.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from functools import partial

    return (partial,)


@app.cell(hide_code=True)
def _():
    ## Export

    from dicekit import Dice, Weight, p
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from collections import defaultdict
    from dataclasses import dataclass
    from functools import reduce
    from itertools import product
    from operator import mul
    from typing import Any, cast


    @dataclass(frozen=True)
    class Event:
        node: "Node"
        value: object

        def __and__(self, other: object) -> "Evidence":
            return Evidence.from_items(self, other)


    @dataclass(frozen=True)
    class Evidence:
        events: frozenset[Event]

        @classmethod
        def from_items(cls, *items: object) -> "Evidence":
            events = []
            for item in items:
                if isinstance(item, Evidence):
                    events.extend(item.events)
                elif isinstance(item, Event):
                    events.append(item)
                else:
                    raise TypeError("evidence must contain events")

            by_node = {}
            for event in events:
                previous = by_node.get(event.node)
                if previous is not None and previous != event.value:
                    raise ValueError(f"conflicting evidence for {event.node.name!r}")
                by_node[event.node] = event.value
            return cls(frozenset(Event(node, value) for node, value in by_node.items()))

        def __and__(self, other: object) -> "Evidence":
            return Evidence.from_items(self, other)

        def as_dict(self) -> dict[object, object]:
            return {event.node: event.value for event in self.events}


    class Node:
        def __init__(self, dag: "DAG", name: str, dist: object, parents: Sequence["Node"] = ()) -> None:
            self.dag = dag
            self.name = name
            self.dist = dist
            self.parents = tuple(parents)
            self._dist_cache = {}

        def __repr__(self) -> str:
            return f"Node({self.name!r})"

        def __eq__(self, value: object) -> Event:  # type: ignore[override]
            return Event(self, value)

        def __hash__(self) -> int:
            return id(self)

        def parent_assignments(self) -> "Iterable[tuple[object, ...]]":
            if not self.parents:
                return [()]
            return product(*[parent.outcomes() for parent in self.parents])

        def dice_for(self, assignment: Sequence[object] = ()) -> "Dice[Any]":
            if not self.parents:
                if assignment:
                    raise ValueError("root nodes do not accept parent assignments")
                key = ()
            else:
                key = tuple(assignment)

            if key not in self._dist_cache:
                if self.parents:
                    dice_like = cast(Callable[..., object], self.dist)(*key)
                else:
                    dice_like = self.dist
                self._dist_cache[key] = _coerce_dice(dice_like)
            return self._dist_cache[key]

        def outcomes(self) -> tuple[object, ...]:
            values = set()
            for assignment in self.parent_assignments():
                values.update(self.dice_for(cast(Sequence[object], assignment)).probs)
            return tuple(values)


    class Factor:
        def __init__(self, variables: Sequence[Node], probs: Mapping[tuple[object, ...], Weight]) -> None:
            self.variables = tuple(variables)
            self.probs = dict(probs)

        def restrict(self, evidence: object) -> "Factor":
            evidence = _normalize_evidence(evidence)
            kept_variables = tuple(var for var in self.variables if var not in evidence)
            kept_probs: defaultdict[tuple[object, ...], Weight] = defaultdict(int)

            for key, probability in self.probs.items():
                row = dict(zip(self.variables, key))
                if all(
                    node not in row or row[cast(Node, node)] == value
                    for node, value in evidence.items()
                ):
                    kept_key = tuple(row[var] for var in kept_variables)
                    kept_probs[kept_key] += probability
            return Factor(kept_variables, kept_probs)

        def sum_out(self, variable: Node) -> "Factor":
            kept_variables = tuple(var for var in self.variables if var is not variable)
            summed: defaultdict[tuple[object, ...], Weight] = defaultdict(int)
            for key, probability in self.probs.items():
                row = dict(zip(self.variables, key))
                kept_key = tuple(row[var] for var in kept_variables)
                summed[kept_key] += probability
            return Factor(kept_variables, summed)

        def normalize(self) -> "Factor":
            total = sum(self.probs.values())
            if total == 0:
                raise ValueError("evidence has zero probability")
            return Factor(self.variables, {key: value / total for key, value in self.probs.items()})

        def __mul__(self, other: "Factor") -> "Factor":
            variables = tuple(dict.fromkeys((*self.variables, *other.variables)))
            joined: defaultdict[tuple[object, ...], Weight] = defaultdict(int)

            for left_key, left_probability in self.probs.items():
                left = dict(zip(self.variables, left_key))
                for right_key, right_probability in other.probs.items():
                    right = dict(zip(other.variables, right_key))
                    shared = set(self.variables) & set(other.variables)
                    if all(left[node] == right[node] for node in shared):
                        row = {**left, **right}
                        key = tuple(row[var] for var in variables)
                        joined[key] += left_probability * right_probability
            return Factor(variables, joined)

        def to_dice(self, node: Node) -> "Dice[Any]":
            if self.variables != (node,):
                raise ValueError("factor must contain exactly the requested node")
            return Dice({key[0]: probability for key, probability in self.probs.items()})


    class DAG:
        def __init__(self) -> None:
            self._nodes: dict[str, Node] = {}

        def add_variable(self, name: str, dist: object, given: object = None) -> Node:
            if name in self._nodes:
                raise ValueError(f"node {name!r} already exists")
            parents = self._normalize_parents(given)
            for parent in parents:
                if parent.dag is not self:
                    raise ValueError("parents must belong to this DAG")
            node = Node(self, name, dist, parents)
            self._nodes[name] = node
            return node

        def var(self, name: str, dist: object, given: object = None) -> Node:
            return self.add_variable(name, dist, given=given)

        def _normalize_parents(self, given: object) -> tuple[Node, ...]:
            if given is None:
                return ()
            if isinstance(given, Node):
                return (given,)
            if isinstance(given, str):
                return (self._nodes[given],)
            return tuple(
                self._nodes[parent] if isinstance(parent, str) else cast(Node, parent)
                for parent in cast(Sequence[object], given)
            )

        def variable(self, name: str) -> Node:
            return self._nodes[name]

        def get_variable(self, name: str) -> Node:
            return self.variable(name)

        def get_variables(self, *names: str) -> Node | tuple[Node, ...]:
            normalized_names = _normalize_names(*names)
            found = tuple(self.variable(name) for name in normalized_names)
            return found[0] if len(found) == 1 else found

        def variables(self, *names: str) -> Node | tuple[Node, ...]:
            return self.get_variables(*names)

        def nodes(self, *names: str) -> Node | tuple[Node, ...]:
            return self.get_variables(*names)

        def factor(self, node: Node) -> Factor:
            variables = (*node.parents, node)
            rows = {}
            for assignment in node.parent_assignments():
                assignment = tuple(cast(Sequence[object], assignment))
                dice = node.dice_for(assignment)
                for value, probability in dice.probs.items():
                    rows[(*assignment, value)] = probability
            return Factor(variables, rows)

        def query(self, target: Node | Event, given: object = None) -> "Dice[Any] | Weight":
            evidence = _normalize_evidence(given)
            event = target if isinstance(target, Event) else None
            target_node = event.node if event else target
            target_node = cast(Node, target_node)
            _validate_same_dag(self, target_node, evidence)

            factors = [self.factor(node).restrict(evidence) for node in self._nodes.values()]
            result = reduce(mul, factors)

            keep = {target_node}
            for variable in list(result.variables):
                if variable not in keep:
                    result = result.sum_out(variable)

            result = result.normalize()

            if event:
                dice = result.to_dice(target_node)
                return dice.probs.get(event.value, 0)
            return result.to_dice(target_node)


    def _coerce_dice(value: object) -> "Dice[Any]":
        if isinstance(value, Dice):
            return value
        return Dice(cast(Mapping[Any, Weight], value))


    def _normalize_names(*names: str) -> tuple[str, ...]:
        if len(names) == 1 and isinstance(names[0], str):
            return tuple(name.strip() for name in names[0].split(",") if name.strip())
        return names


    def _normalize_evidence(given: object) -> dict[object, object]:
        if given is None:
            return {}
        if isinstance(given, dict):
            return given
        if isinstance(given, Event):
            return {given.node: given.value}
        if isinstance(given, Evidence):
            return given.as_dict()
        raise TypeError("given must be an event or evidence set")


    def _validate_same_dag(dag: DAG, target_node: Node, evidence: Mapping[object, object]) -> None:
        nodes = [target_node, *evidence]
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError("query targets and evidence must use DAG nodes")
            if node.dag is not dag:
                raise ValueError("query cannot mix nodes from different DAGs")


    @p.register(Node)
    def _p_node(expression: "Node", given: object = None) -> object:
        return expression.dag.query(expression, given=given)


    @p.register(Event)
    def _p_event(expression: "Event", given: object = None) -> object:
        return expression.node.dag.query(expression, given=given)


    # The puzzle API now lives in core dicekit: a bare `Dice` created inside
    # `model()` is itself a variable, and `Dice.from_function(fn, parents=[...])`
    # adds a dependent one. Re-export the model helpers for convenience.
    from dicekit import World, model  # noqa: F401

    return DAG, Dice, model, p


@app.cell
def _(DAG, partial):
    dag = DAG()

    dag.add_variable("disease", {"present": 0.01, "absent": 0.99})

    def test_result(disease, sensitivity, false_positive_rate):
        if disease == "present":
            return {
                "positive": sensitivity,
                "negative": 1 - sensitivity,
            }
        return {
            "positive": false_positive_rate,
            "negative": 1 - false_positive_rate,
        }

    tests = [
        ("test_1", 0.90, 0.05),
        ("test_2", 0.90, 0.05),
        ("test_3", 0.90, 0.05),
    ]

    for name, sensitivity, false_positive_rate in tests:
        dag.add_variable(
            name,
            given="disease",
            dist=partial(
                test_result,
                sensitivity=sensitivity,
                false_positive_rate=false_positive_rate,
            )
        )

    disease, test_1, test_2, test_3 = dag.get_variables("disease, test_1, test_2, test_3")

    disease, test_1, test_2, test_3
    return disease, test_1, test_2, test_3


@app.cell
def _(disease, mo, p, test_1, test_2, test_3):
    scenarios = [
        ("0 observed positives", None),
        ("1 observed positive", test_1 == "positive"),
        (
            "2 observed positives",
            (test_1 == "positive") & (test_2 == "positive"),
        ),
        (
            "3 observed positives",
            (test_1 == "positive") & (test_2 == "positive") & (test_3 == "positive"),
        ),
    ]

    mo.hstack([p(disease, given=evidence) for label, evidence in scenarios])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same inference, draggable

    The block above reads `p(disease, given=…)` off for a few *fixed* evidence
    scenarios. Here is the same disease/test model, built with core `model()`.
    `clinic.show()` is the whole UI: click **edit** on any node and drag its
    bars to assert a belief on that variable — every other node repaints in
    place, parents included (that's `Dice.believe()` under the hood; **reset**
    is `Dice.forget()`). Drag a test toward `"positive"` and the `disease`
    posterior climbs; pin all three fully positive and you reproduce the
    "3 observed positives" number above. Several dragged beliefs are satisfied
    *jointly*, so the order you drag them in doesn't matter.
    """)
    return


@app.cell
def _(Dice, model):
    def test_result_core(d, sensitivity=0.90, false_positive_rate=0.05):
        if d == "present":
            return {"positive": sensitivity, "negative": 1 - sensitivity}
        return {"positive": false_positive_rate, "negative": 1 - false_positive_rate}

    with model() as clinic:
        disease_v = Dice({"present": 0.01, "absent": 0.99})
        test_a = Dice.from_function(test_result_core, parents=[disease_v])
        test_b = Dice.from_function(test_result_core, parents=[disease_v])
        test_c = Dice.from_function(test_result_core, parents=[disease_v])
    return clinic, disease_v


@app.cell
def _(clinic):
    clinic.show()
    return


@app.cell
def _(disease_v, p):
    # beliefs dragged into the graph above are world state, not view state:
    # re-run this cell after dragging and the posterior reflects them.
    p(disease_v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Looking inside: the factor graph

    The model above is a Bayes net, and its joint factorizes as
    $P(\text{disease}) \cdot \prod_i P(\text{test}_i \mid \text{disease})$.
    The squares below are exactly those factors — the prior and one CPT
    heatmap per test — drawn between the variables they connect. A pinned
    belief is *just one more (unary) factor* plugged into its variable:
    drag a node in the graph above (or here — these charts are live too)
    and watch its belief factor switch on; `reset` and it vanishes.
    """)
    return


@app.cell
def _(clinic, mo):
    # factor-graph view (prototype, notebook-only): reuses show()'s private
    # wiring hooks on purpose — promote into the package once it feels right.
    from itertools import product as _product

    from wigglystuff import ObservablePlot as _ObservablePlot
    from wigglystuff import WidgetDAG as _WidgetDAG

    _CPT_JS = """Plot.plot({
        width, height,
        marginLeft: 56, marginBottom: 26,
        x: {label: null}, y: {label: null},
        color: {domain: [0, 1], scheme: "blues"},
        marks: [
            Plot.cell(data, {x: "out", y: "given", fill: "p"}),
            Plot.text(data, {x: "out", y: "given",
                text: (d) => d.p.toFixed(2),
                fill: (d) => (d.p > 0.55 ? "white" : "black")}),
        ],
    })"""

    _BELIEF_JS = """
    if (!active) {
        const div = document.createElement("div");
        div.style.cssText = "width:" + width + "px;height:" + height + "px;" +
            "display:flex;align-items:center;justify-content:center;" +
            "color:#bbb;font:10px sans-serif;border:1px dashed #ccc;border-radius:4px";
        div.textContent = "no belief";
        return div;
    }
    return Plot.plot({
        width, height,
        marginLeft: 36, marginBottom: 22,
        y: {domain: [0, 1], label: null}, x: {label: null},
        marks: [Plot.barY(data, {x: "i", y: "p", fill: "#ff725c"}), Plot.ruleY([0])],
    });
    """

    def _cpt_chart(node):
        # the conditional P(node | parents) as a heatmap: one row per parent
        # assignment, one column per outcome. CPTs are the model, so static.
        rows = []
        for values in _product(*[list(parent.probs) for parent in node._parents]):
            dist = node._fn(*values)
            dist = dist.probs if hasattr(dist, "probs") else dist
            total = sum(dist.values())
            for out, weight in dist.items():
                rows.append({
                    "given": " · ".join(map(str, values)),
                    "out": str(out),
                    "p": weight / total,
                })
        return _ObservablePlot(_CPT_JS, variables={"data": rows}, width=180, height=100)

    def _prior_chart(node):
        rows = [{"given": "p", "out": str(out), "p": w} for out, w in node.probs.items()]
        return _ObservablePlot(_CPT_JS, variables={"data": rows}, width=180, height=80)

    _entries = clinic._belief_entries()
    _var_names = {id(node): (node._name or f"var{i}") for i, node in enumerate(clinic.sources)}

    _fg_nodes = {}
    _fg_edges = []
    for _node, _plot, _outs in _entries:
        _fg_nodes[_var_names[id(_node)]] = mo.ui.anywidget(_plot)

    _slots = {}
    for _node in clinic.sources:
        _label = _var_names[id(_node)]
        if _node._parents:
            _factor = f"p({_label}|…)"
            _fg_nodes[_factor] = _cpt_chart(_node)
            for _parent in _node._parents:
                _fg_edges.append((_var_names[id(_parent)], _factor))
        else:
            _factor = f"p({_label})"
            _fg_nodes[_factor] = _prior_chart(_node)
        _fg_edges.append((_factor, _label))

        _slot = _ObservablePlot(_BELIEF_JS, variables={"data": [], "active": False}, width=150, height=90)
        _slots[id(_node)] = _slot
        _fg_nodes[f"belief({_label})"] = _slot
        _fg_edges.append((f"belief({_label})", _label))

    def _refresh_slots(_origin=None):
        for _n in clinic.sources:
            _belief = clinic._beliefs.get(_n)
            _total = sum(_belief.values()) if _belief else 1
            _slots[id(_n)].variables = {
                **_slots[id(_n)].variables,
                "active": _belief is not None,
                "data": [{"i": str(k), "p": v / _total} for k, v in (_belief or {}).items()],
            }

    _refresh_slots()
    # re-running this cell stacks a stale listener; harmless for a prototype
    clinic._belief_listeners.append(_refresh_slots)
    _WidgetDAG(_fg_nodes, _fg_edges)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ...and the joint, atom by atom

    Under the hood nothing but this table exists: $2 \cdot 2^3 = 16$ atoms,
    one per full assignment, weighted by the product of the factors above.
    Grey bars are the prior joint; blue bars are the joint under the current
    beliefs. Conditioning is *literally* reweighting this table: pin a test
    positive above and watch the `−` atoms die and the `sick` atoms grow
    (square-root scale, because the healthy-all-negative atom dominates).
    """)
    return


@app.cell
def _(clinic):
    from dicekit import _jeffrey_condition as _reweight
    from wigglystuff import ObservablePlot as _AtomsPlot

    _ATOMS_JS = """Plot.plot({
        width, height,
        marginLeft: 44, marginBottom: 66,
        x: {label: null, tickRotate: -45, domain: data.map((d) => d.label)},
        y: {label: "p(atom)", type: "sqrt", grid: true},
        marks: [
            Plot.barY(data, {x: "label", y: "prior", fill: "#d5d5d5"}),
            Plot.barY(data, {x: "label", y: "post", fill: "#4269d0", opacity: 0.8}),
            Plot.ruleY([0]),
        ],
    })"""

    _short = {"present": "sick", "absent": "ok", "positive": "+", "negative": "−"}

    _base_rows = clinic.joint()
    _prior_total = sum(w for _, w in _base_rows)
    _labels = [
        " ".join(_short.get(a[s], str(a[s])) for s in clinic.sources)
        for a, _ in _base_rows
    ]
    _priors = [w / _prior_total for _, w in _base_rows]

    _atoms = _AtomsPlot(_ATOMS_JS, variables={"data": []}, width=640, height=230)

    def _refresh_atoms(_origin=None):
        _rows = _reweight(list(_base_rows), dict(clinic._beliefs))
        _t = sum(w for _, w in _rows) or 1
        _post = {id(a): w / _t for a, w in _rows}
        _atoms.variables = {
            **_atoms.variables,
            "data": [
                {"label": _labels[_i], "prior": _priors[_i], "post": _post.get(id(_a), 0)}
                for _i, (_a, _w) in enumerate(_base_rows)
            ],
        }

    _refresh_atoms()
    clinic._belief_listeners.append(_refresh_atoms)
    _atoms
    return


@app.cell
def _(p, test_1, test_2):
    p(test_1, given=test_2 == "positive")
    return


@app.cell
def _(p, test_1, test_2):
    p(test_1 == "positive", given=test_2 == "positive")
    return


@app.cell
def _(Dice, model):
    # from dicekit import Dice, model, p

    with model() as m:
        car    = Dice.from_numbers(1, 2, 3)              # bare Dice == a root variable
        pick   = Dice.from_numbers(1, 2, 3)
        opened = Dice.from_function(                      # only dependence needs a constructor
            lambda c, g: Dice.from_numbers(*[d for d in (1,2,3) if d != c and d != g]),
            parents=[car, pick],                         # explicit list, per your preference
        )
        switch = Dice.from_numbers(True, False)
        win    = (switch & (car != pick)) | (~switch & (car == pick))


    m.show()
    return (car,)


@app.cell
def _(car):
    # a model variable displays as a live node: it repaints when beliefs
    # change (drag in the graph above) and has its own edit toggle.
    # (`car.marginal()` would be a static snapshot Dice instead.)
    car.marginal()
    return


if __name__ == "__main__":
    app.run()
