__all__ = ['Weight', 'Dice', 'Factor', 'World', 'model', 'Vase', 'p', 'exp', 'var', 'mix', 'ordered']


import marimo as mo
from wigglystuff import ObservablePlot
import random
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product, permutations
from functools import reduce
from fractions import Fraction
from operator import mul
from typing import Any, Self, TypeAlias, cast
import contextvars
import dis
import operator
import sys

# Probability weights may be exact (Fraction) or inexact (int/float). Fraction
# interoperates cleanly with int and float (Fraction + float -> float), so mixing
# downgrades to float rather than erroring. Decimal is intentionally excluded
# because Decimal + float raises at runtime.
Weight: TypeAlias = int | float | Fraction


def _plot_value(value: Any) -> Any:
    """
    Coerce a value into something a chart can serialize.

    Numeric outcomes/probabilities (Fraction, Decimal, numpy scalars, ...) are
    cast to float so they render, while non-numeric keys such as the string keys
    produced by a Vase (and boolean comparison outcomes) are left untouched.
    """
    if isinstance(value, (bool, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# The one bar-chart-with-drag JS, parameterized by `uiChrome`:
# - "editor" (Dice.editor): always draggable, with a `reset` link.
# - "belief" (World.show nodes): renders as a plain posterior chart with an
#   `edit` toggle; while editing the bars drag and commit as a *belief*
#   (`touched` in the synced variables), `reset` withdraws it. All state
#   that must survive re-renders rides in the `variables` trait, because
#   the widget fully re-runs this code on every variables change — its own
#   commits and the kernel's posterior pushes alike.
_EDITOR_JS = """
const n = data.length;
const mode = typeof editMode === "string" ? editMode : "renormalize";
const chrome = typeof uiChrome === "string" ? uiChrome : "editor";
const state = model.get("variables") || {};
let touched = !!state.touched;
let editing = chrome === "editor" ? true : !!state.editing;
const margin = {top: 20, right: 12, bottom: 30, left: 44};
const iw = Math.max(10, width - margin.left - margin.right);
const ih = Math.max(10, height - margin.top - margin.bottom);

const svg = d3.create("svg")
    .attr("width", width).attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .style("max-width", "100%").style("font", "10px sans-serif")
    // stop dragging from turning into a text selection of the labels
    .style("user-select", "none")
    .style("-webkit-user-select", "none")
    .style("touch-action", "none");

const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

const x = d3.scaleBand()
    .domain(d3.range(n).map((i) => String(data[i].i)))
    .range([0, iw]).padding(0.15);
const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

g.append("g").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x));
g.append("g").call(d3.axisLeft(y).ticks(5, "%"));

// `probs`, not `weights`: the injected `variables` keys arrive as
// function parameters, so a `let weights` would redeclare one.
const original = data.map((d) => Number(d.p));
let probs = original.slice();

// visible bars — display only, so they never steal the drag
const bars = g.selectAll("rect.bar").data(d3.range(n)).join("rect")
    .attr("class", "bar")
    .attr("x", (i) => x(String(data[i].i)))
    .attr("width", x.bandwidth())
    .attr("pointer-events", "none");

const labels = g.selectAll("text.lab").data(d3.range(n)).join("text")
    .attr("class", "lab")
    .attr("x", (i) => x(String(data[i].i)) + x.bandwidth() / 2)
    .attr("text-anchor", "middle").attr("font-size", 9)
    .attr("pointer-events", "none");

// full-height transparent targets: a bar dragged to 0 is still grabbable
const hits = g.selectAll("rect.hit").data(d3.range(n)).join("rect")
    .attr("class", "hit")
    .attr("x", (i) => x(String(data[i].i)))
    .attr("width", x.bandwidth())
    .attr("y", 0).attr("height", ih)
    .attr("fill", "transparent")
    .style("cursor", editing ? "ns-resize" : "default");

function persist(extra) {
    model.set("variables", {...model.get("variables"), ...extra});
    model.save_changes();
}

function redraw() {
    bars.attr("y", (i) => y(probs[i])).attr("height", (i) => ih - y(probs[i]))
        // color signals *mode*: static charts stay black (like plain
        // prob charts), a node in edit mode turns blue; an asserted
        // belief is indicated by its `reset` link, not a color
        .attr("fill", editing ? "#4269d0" : "currentColor");
    labels.attr("y", (i) => Math.max(9, y(probs[i]) - 3))
        .text((i) => probs[i].toFixed(2));
}
redraw();

function setBar(idx, target) {
    target = Math.max(0, Math.min(1, target));
    if (mode === "locked") {
        const others = probs.reduce((a, w, j) => a + (j === idx ? 0 : w), 0);
        const room = 1 - target;
        if (others <= 0) {
            const share = room / Math.max(1, n - 1);
            probs = probs.map((w, j) => (j === idx ? target : share));
        } else {
            const scale = room / others;
            probs = probs.map((w, j) => (j === idx ? target : w * scale));
        }
    } else {
        probs[idx] = target;
    }
    redraw();
}

function commit() {
    if (mode !== "locked") {
        const total = probs.reduce((a, b) => a + b, 0);
        if (total > 0) probs = probs.map((w) => w / total);
        redraw();
    }
    persist({
        data: d3.range(n).map((i) => ({i: data[i].i, p: probs[i]})),
        weights: probs.slice(),
        faces: data.map((d) => d.i),
        touched: touched,
        editing: editing,
    });
}

function reset() {
    probs = original.slice();
    touched = false;
    // in a graph, reset is also the way *out* of edit mode: it withdraws
    // the belief and hands the node back to the live flow. A separate
    // "done" reads as that but would keep the belief pinned — confusing.
    if (chrome === "belief") editing = false;
    redraw();
    commit();
}

// control links: a live node offers `edit`, an editing node only `reset`
const controls = [];
if (chrome === "belief" && !editing) {
    controls.push({label: "edit", act: () => persist({editing: true})});
} else {
    controls.push({label: "reset", act: reset});
}
controls.forEach((ctrl, index) => {
    svg.append("text")
        .attr("x", width - margin.right - 40 * index).attr("y", 14)
        .attr("text-anchor", "end").attr("fill", "#4269d0")
        .style("cursor", "pointer").style("text-decoration", "underline")
        .text(ctrl.label)
        .on("click", (event) => { event.stopPropagation(); ctrl.act(); });
});

if (editing) {
    const drag = d3.drag()
        .container(() => g.node())
        .on("start", (event) => { if (event.sourceEvent) event.sourceEvent.preventDefault(); })
        .on("drag", (event, i) => { touched = true; setBar(i, y.invert(event.y)); })
        .on("end", commit);
    hits.call(drag);
}

return svg.node();
"""


class Dice[T: Hashable]:
    """
    A class representing dice with arbitrary probability distributions.

    This class allows creating, manipulating, and visualizing dice with
    custom probability distributions for each face value.
    """

    def __init__(self, probs: Mapping[T, Weight]) -> None:
        """
        Initialize a Dice object with given probabilities.

        Parameters:
            probs (dict): A dictionary mapping face values to probabilities

        When constructed inside a ``with model():`` block, the dice also
        registers itself as a *root variable* of that model (a column of
        the shared joint), capturing the Python name it is assigned to.
        Outside a model it is an ordinary, standalone distribution.
        """
        total = sum(probs.values())
        self._probs: "dict[Any, Weight] | None" = {k: v / total for k, v in probs.items()}
        self._world: "World | None" = None
        self._op: "Callable[..., Any] | None" = None
        self._operands: tuple[Any, ...] = ()
        self._parents: tuple[Any, ...] = ()
        self._fn: "Callable[..., Any] | None" = None
        self._is_predicate = False
        self._is_source = False
        self._name: str | None = None
        self._lazy = False
        self._probs_version = -1
        self._live_plot: Any = None
        self._dist_cache: "dict[tuple[Any, ...], Dice[Any]]" = {}
        self._support_cache: "tuple[Any, ...] | None" = None
        self._node_factor: "Factor | None" = None

        world = _active_world.get()
        if world is not None:
            self._world = world
            self._is_source = True
            self._name = _assigned_name()
            world._register(self)

    @classmethod
    def _make(
        cls,
        *,
        world: "World | None",
        probs: "Mapping[Any, Weight] | None" = None,
        op: "Callable[..., Any] | None" = None,
        operands: "Sequence[Any]" = (),
        parents: "Sequence[Dice[Any]]" = (),
        fn: "Callable[..., Any] | None" = None,
        is_predicate: bool = False,
        is_source: bool = False,
        name: str | None = None,
    ) -> "Dice[Any]":
        """Build a Dice directly, bypassing ``__init__`` registration.

        The single internal factory for both plain dice (``_raw``) and the
        lazy *variable* dice built by ``variable`` and the operators.
        """
        obj = cls.__new__(cls)
        obj._probs = dict(probs) if probs is not None else None
        obj._world = world
        obj._op = op
        obj._operands = tuple(operands)
        obj._parents = tuple(parents)
        obj._fn = fn
        obj._is_predicate = is_predicate
        obj._is_source = is_source
        obj._name = name
        obj._lazy = probs is None
        obj._probs_version = -1
        obj._live_plot = None
        obj._dist_cache = {}
        obj._support_cache = None
        obj._node_factor = None
        return obj

    @classmethod
    def _raw(cls, probs: "Mapping[Any, Weight]") -> "Dice[Any]":
        """A standalone dice from raw weights, never registered in a model.

        Used for every library-internal construction (``operate``, ``map``,
        query results, ...) so intermediate dice never pollute the active
        model's set of variables.
        """
        total = sum(probs.values())
        return cls._make(world=None, probs={k: v / total for k, v in probs.items()})

    @property
    def probs(self) -> "dict[Any, Weight]":
        """The outcome→probability mapping.

        For a lazy *variable* (a dependent or a derived expression) this is
        the marginal over the model's joint under the world's current
        beliefs, cached until those beliefs change. A root's ``probs`` stays
        its prior — that is its generative definition; read a root's
        posterior with :meth:`marginal` or ``p()``.
        """
        if self._lazy:
            world = self._world
            assert world is not None
            if self._probs is None or self._probs_version != world._belief_version:
                self._probs = world.query(self)._probs
                self._probs_version = world._belief_version
        return cast("dict[Any, Weight]", self._probs)

    def __hash__(self) -> int:
        # A Dice is identity-hashed: __eq__ builds a new Dice rather than a
        # bool, so value-equality is unavailable, and model variables must
        # be usable as dict keys (columns of the joint).
        return id(self)

    @classmethod
    def from_sides(cls, n: int = 6) -> "Dice[int]":
        """
        Create a fair dice with n sides.

        Parameters:
            n (int): Number of sides, default is 6

        Returns:
            Dice: A fair dice with n sides
        """
        return cast(Dice[int], cast(Any, cls)({i: 1 / n for i in range(1, n + 1)}))

    @classmethod
    def from_numbers[U: Hashable](cls, *args: U) -> "Dice[U]":
        """
        Create a dice from a sequence of numbers with frequencies.

        Parameters:
            *args: Variable length list of numbers

        Returns:
            Dice: A dice with probabilities based on the frequency of each number
        """
        c = Counter(args)
        return cast(Dice[U], cast(Any, cls)({k: v / len(args) for k, v in c.items()}))

    @classmethod
    def from_dist(
        cls,
        dist: Any,
        n: int = 6,
        quantiles: Sequence[float] | None = None,
    ) -> "Dice[float]":
        """
        Approximate a distribution with a dice by slicing equiprobable quantiles.

        `dist` is any object exposing an inverse-CDF: a frozen scipy.stats
        distribution (``.ppf``) or a ``statistics.NormalDist`` (``.inv_cdf``).

        By default the dice has `n` equiprobable faces placed at the midpoint
        of each quantile bin, i.e. ``ppf((i + 0.5) / n)`` for ``i`` in
        ``range(n)``. Pass an explicit `quantiles` sequence of probabilities in
        (0, 1) to control where the slices are taken; each resulting face is
        still equiprobable.

        Parameters:
            dist: An object with a `.ppf` or `.inv_cdf` inverse-CDF method
            n (int): Number of equiprobable faces, default is 6
            quantiles: Optional sequence of probabilities in (0, 1) to slice at

        Returns:
            Dice: A dice approximating the distribution
        """
        if hasattr(dist, "ppf"):
            inv_cdf = dist.ppf
        elif hasattr(dist, "inv_cdf"):
            inv_cdf = dist.inv_cdf
        else:
            raise TypeError("from_dist needs an object with .ppf or .inv_cdf")

        qs = list(quantiles) if quantiles is not None else [(i + 0.5) / n for i in range(n)]
        if not qs:
            raise ValueError("from_dist needs at least one quantile")
        if any(not (0 < q < 1) for q in qs):
            raise ValueError("quantiles must be strictly between 0 and 1")

        probs: dict[float, Weight] = {}
        for q in qs:
            outcome = float(inv_cdf(q))
            probs[outcome] = probs.get(outcome, 0) + 1 / len(qs)
        return cast("Dice[float]", cast(Any, cls)(probs))

    @classmethod
    def from_weights(cls, faces: "Sequence[Any]", weights: "Sequence[Weight]") -> "Dice[Any]":
        """Build a dice from parallel ``faces`` and ``weights``.

        Weights are normalized to sum to 1. Like the other public
        constructors (``from_numbers``, ``from_dist``), the dice registers as
        a *source* variable when created inside a ``with model():`` block, so
        an edited prior can drive a model. This is the reconstruction path for
        :meth:`editor`: the widget hands back edited ``faces``/``weights`` (as
        plain JSON) and this rebuilds a ``Dice``.
        """
        return cast("Dice[Any]", cast(Any, cls)({f: w for f, w in zip(faces, weights)}))

    @classmethod
    def from_editor(cls, widget: Any) -> "Dice[Any]":
        """Reconstruct a dice from the state of an :meth:`editor` widget.

        Accepts either the ``ObservablePlot`` returned by :meth:`editor` or a
        ``marimo.ui.anywidget`` wrapping it, reads the edited ``faces`` and
        ``weights`` out of its ``variables`` trait, and returns a normalized
        ``Dice``. Falls back to the untouched distribution if the widget has
        not been dragged yet.
        """
        variables = getattr(widget, "variables", None)
        if variables is None:
            value = getattr(widget, "value", {}) or {}
            variables = value.get("variables", value)
        faces = variables.get("faces")
        weights = variables.get("weights")
        if faces is None or weights is None:
            data = variables.get("data", [])
            faces = [row["i"] for row in data]
            weights = [row["p"] for row in data]
        return cls.from_weights(faces, weights)

    @classmethod
    def from_function(cls, fn: "Callable[..., Any]", parents: "Sequence[Dice[Any]]" = ()) -> "Dice[Any]":
        """Create a *dependent* variable from a conditional distribution.

        ``fn`` receives the values of ``parents`` (in order) and returns a
        Dice or mapping — the conditional ``P(this | parents)``. This is the
        one constructor for dependence inside a ``model()``; a *root*
        variable needs no function, it is just a bare ``Dice``.

            opened = Dice.from_function(
                lambda car, pick: Dice.from_numbers(*doors_left(car, pick)),
                parents=[car, pick],
            )
        """
        if not callable(fn):
            raise TypeError("Dice.from_function needs a conditional-distribution function; a root variable is just a bare Dice")
        parents = list(parents)
        if not parents:
            raise TypeError("Dice.from_function needs parents; a root variable is just a bare Dice inside a model")
        if not all(isinstance(parent, Dice) and parent._world is not None for parent in parents):
            raise TypeError("parents must be model variables")
        world = parents[0]._world
        for parent in parents:
            if parent._world is not world:
                raise ValueError("parents must belong to the same model")
        assert world is not None
        node = cls._make(world=world, parents=parents, fn=fn, is_source=True, name=_assigned_name())
        world._register(node)
        return node

    def roll(self, n: int = 1) -> list[T]:
        """
        Simulate rolling the dice n times.

        Parameters:
            n (int): Number of rolls, default is 1

        Returns:
            list: Results of the dice rolls
        """
        return random.choices(
            list(self.probs.keys()),
            weights=cast(Any, list(self.probs.values())),
            k=n,
        )

    def sample(self, n: int = 1) -> list[T]:
        """
        Simulate rolling the dice n times.

        Parameters:
            n (int): Number of rolls, default is 1

        Returns:
            list: Results of the dice rolls
        """
        return self.roll(n=1)

    def map[U: Hashable](self, transform: Callable[[T], U] | Mapping[T, U]) -> "Dice[U]":
        """
        Transform the outcomes of the dice.

        Parameters:
            transform: A callable or mapping from old outcomes to new outcomes

        Returns:
            Dice: A dice containing the transformed outcomes
        """
        func = transform.__getitem__ if isinstance(transform, Mapping) else transform
        new_probs = {}
        for outcome, probability in self.probs.items():
            new_outcome = func(outcome)
            new_probs[new_outcome] = new_probs.get(new_outcome, 0) + probability
        return Dice._raw(new_probs)

    def operate[U: Hashable, V: Hashable](
        self,
        other: "Dice[U] | U",
        operator: Callable[[T, U], V],
    ) -> "Dice[V]":
        """
        Apply an operation between this dice and another dice or number.

        Parameters:
            other: Another Dice object or a number
            operator: A function that takes two arguments and returns a result

        Returns:
            Dice: A new dice representing the distribution of the operation's results
        """
        if not isinstance(other, Dice):
            try:
                other = Dice._raw({other: 1})
            except TypeError as exc:
                raise TypeError("operations require a Dice or hashable value") from exc
        other = cast(Dice[U], other)
        new_probs: dict[V, Weight] = {}
        for s1, p1 in self.probs.items():
            for s2, p2 in other.probs.items():
                new_key = operator(s1, s2)
                if new_key not in new_probs:
                    new_probs[new_key] = 0
                new_probs[new_key] += p1 * p2
        return Dice._raw(new_probs)

    def filter(self, func: Callable[[T], bool]) -> "Dice[T]":
        """
        Create a new dice by filtering face values based on a function.

        Parameters:
            func: A function that returns True for values to keep

        Returns:
            Dice: A new dice with only the filtered values, renormalized
        """
        new_probs = {k: v for k, v in self.probs.items() if func(k)}
        total_prob = sum(new_probs.values())
        return Dice._raw({k: v / total_prob for k, v in new_probs.items()})

    def _live_chart(self) -> Any:
        """This variable's wired belief chart — the same node ``World.show``
        renders, created once and reused so repeated displays share one
        widget (and one belief listener) instead of stacking them up.
        """
        assert self._world is not None
        if self._live_plot is None:
            [(_, plot, _)] = self._world._belief_entries(nodes=[self])
            self._live_plot = plot
        return self._live_plot

    def _display_(self) -> Any:
        """Render for marimo, which prefers ``_display_`` over mimebundles.

        Returning the *widget object* (rather than its mimebundle) lets
        marimo wrap it via ``mo.ui.anywidget``, which keeps the comm live —
        a mimebundle renders as a static snapshot. A model variable
        displays as its live belief chart: it repaints when the model's
        beliefs change (e.g. drags in :meth:`World.show`) and carries its
        own ``edit`` toggle. A standalone dice displays its pmf chart.
        """
        if self._world is not None:
            return self._live_chart()
        return self.prob_chart()

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        """
        Render the dice as a live Observable Plot widget in notebooks.

        Jupyter's display path (marimo prefers :meth:`_display_`): the same
        live belief chart for a model variable, the pmf chart for a
        standalone dice.
        """
        return self._display_()._repr_mimebundle_(**kwargs)

    def prob_chart(self) -> Any:
        """
        Create a visualization of the dice probability distribution.

        Returns:
            ObservablePlot: An Observable Plot widget showing the probability
            mass function.
        """
        data = [{"i": _plot_value(k), "p": _plot_value(v)} for k, v in self.probs.items()]
        return ObservablePlot(
            """Plot.plot({
                width, height,
                marginLeft: 44,
                marginBottom: 28,
                x: {
                    label: null,
                    tickFormat: (d) => typeof d === "number"
                        ? (Number.isInteger(d) ? String(d) : d.toFixed(2)) : d,
                    ticks: data.length > 10
                        ? data.map((d) => d.i).filter((_, i) => i % Math.ceil(data.length / 10) === 0)
                        : undefined,
                },
                y: {label: "p", grid: true},
                marks: [
                    Plot.barY(data, {x: "i", y: "p"}),
                    Plot.ruleY([0]),
                ],
            })""",
            variables={"data": data},
            width=220,
            height=180,
        )

    def cdf_chart(self) -> Any:
        """
        Create a visualization of the cumulative distribution function.

        Returns:
            ObservablePlot: An Observable Plot step chart showing the
            cumulative probability.
        """
        cumulative = 0
        data = []
        for outcome, probability in sorted(self.probs.items()):
            cumulative += probability
            data.append({"i": _plot_value(outcome), "p": _plot_value(cumulative)})
        return ObservablePlot(
            """Plot.plot({
                width, height,
                marginLeft: 44,
                marginBottom: 28,
                title: "Cumulative distribution:",
                x: {
                    label: null,
                    tickFormat: (d) => typeof d === "number"
                        ? (Number.isInteger(d) ? String(d) : d.toFixed(2)) : d,
                    ticks: data.length > 10
                        ? data.map((d) => d.i).filter((_, i) => i % Math.ceil(data.length / 10) === 0)
                        : undefined,
                },
                y: {domain: [0, 1], label: "p", grid: true},
                marks: [
                    Plot.lineY(data, {x: "i", y: "p", curve: "step-after"}),
                    Plot.ruleY([0]),
                ],
            })""",
            variables={"data": data},
            width=220,
            height=180,
        )

    def editor(self, mode: str = "renormalize") -> Any:
        """Return a live, draggable widget for editing this dice's probabilities.

        The returned ``ObservablePlot`` renders one bar per outcome; drag a
        bar up or down to change its probability. On release the edited
        weights are written back into the widget's ``variables`` trait, so
        in marimo you can reconstruct the edited dice reactively::

            w = mo.ui.anywidget(die.editor())
            edited = Dice.from_editor(w)   # recomputed on every drag

        Two interaction variations are available via ``mode``:

        * ``"renormalize"`` (default) — drag any bar freely; on release the
          whole vector is renormalized so the probabilities sum to 1.
        * ``"locked"`` — the total stays pinned at 1 *during* the drag: the
          other bars absorb the change proportionally, for a smoother feel.

        The dice itself is not mutated; use :meth:`from_editor` (or
        :meth:`from_weights`) to obtain the edited distribution.
        """
        if mode not in ("renormalize", "locked"):
            raise ValueError("mode must be 'renormalize' or 'locked'")
        data = [{"i": _plot_value(k), "p": _plot_value(v)} for k, v in self.probs.items()]
        return ObservablePlot(
            _EDITOR_JS,
            variables={
                "data": data,
                "editMode": mode,
                "uiChrome": "editor",
                "faces": [row["i"] for row in data],
                "weights": [row["p"] for row in data],
                "touched": False,
            },
            width=280,
            height=220,
        )

    def ordered(self, n: int, k: int | None = None) -> list["Dice[T]"]:
        """
        Take the dice `n` times and calculate the ordered distributions for it.

        Optionally you can also take `k` of these dice.

        Returns:
            List of ordered dice
        """
        items = [self] * n
        if k:
            return ordered(*items, k=k)
        return ordered(*items)

    def out_of[U: Hashable](self, n: int = 2, func: Callable[[list[T]], U] = max) -> "Dice[U]":
        """
        Create a dice representing the result of applying a function to n rolls.

        Parameters:
            n (int): Number of dice to roll, default is 2
            func: Function to apply to the rolls (e.g., max, min), default is max

        Returns:
            Dice: A new dice representing the distribution of the function's results
        """
        result: dict[U, Weight] = {}
        dice_in = [self] * n
        for _i in product(*[d.probs.items() for d in dice_in]):
            values = [_[0] for _ in _i]
            prob = reduce(lambda a, b: a * b, [_[1] for _ in _i])
            outcome = func(values)
            if outcome not in result:
                result[outcome] = 0
            result[outcome] += prob
        return Dice._raw(result)

    # -- model-variable behavior ---------------------------------------
    # A dice bound to a World (``self._world is not None``) is a random
    # variable: a *source* (a root dist, or a dependent with ``_fn`` over
    # ``_parents``) that is a column of the joint, or a *derived*
    # expression (``_op`` over ``_operands``) evaluated lazily. Standalone
    # dice (``self._world is None``) keep the original independent-product
    # operator semantics untouched.

    def _dist_for(self, assignment: "Mapping[Dice[Any], Any]") -> "Dice[Any]":
        # The conditional distribution of this source given an assignment to
        # its parents. A root's distribution is simply itself. Parents are
        # evaluated, not looked up: a *derived* parent (e.g. ``a + b``) is
        # not a column of the joint, only sources are. Results are memoized
        # per parent-value tuple; `fn` is treated as deterministic.
        if self._fn is None:
            return self
        values = tuple(parent._eval(assignment) for parent in self._parents)
        if values not in self._dist_cache:
            self._dist_cache[values] = _coerce_dice(self._fn(*values))
        return self._dist_cache[values]

    def _leaves(self) -> "tuple[Dice[Any], ...]":
        # The source variables (columns of the joint) this expression is
        # built from — itself, if it *is* a source. Order-stable, deduped
        # by identity (a Dice's `==` builds an expression, so membership
        # tests on lists/tuples of dice must never be used).
        if self._op is None:
            return (self,)
        found: "list[Dice[Any]]" = []
        seen: set[int] = set()

        def walk(node: "Dice[Any]") -> None:
            for operand in node._operands:
                if not isinstance(operand, Dice):
                    continue
                if operand._op is None:
                    if id(operand) not in seen:
                        seen.add(id(operand))
                        found.append(operand)
                else:
                    walk(operand)

        walk(self)
        return tuple(found)

    def _support(self) -> "tuple[Any, ...]":
        # Every outcome this variable can take, order-stable. For a derived
        # expression this ranges over the product of its operands' supports
        # — a superset when operands are correlated, which is harmless
        # (impossible values simply never carry mass).
        if self._support_cache is None:
            if self._op is not None:
                combos = product(
                    *[op._support() if isinstance(op, Dice) else (op,) for op in self._operands]
                )
                self._support_cache = tuple(dict.fromkeys(self._op(*combo) for combo in combos))
            elif self._fn is not None:
                factor = self._family_factor()
                self._support_cache = tuple(dict.fromkeys(key[-1] for key in factor.table))
            else:
                self._support_cache = tuple(cast("dict[Any, Weight]", self._probs))
        return self._support_cache

    def _prior_factor(self) -> "Factor":
        # A root's prior as a unary factor.
        if self._node_factor is None:
            probs = cast("dict[Any, Weight]", self._probs)
            self._node_factor = Factor((self,), {(value,): weight for value, weight in probs.items()})
        return self._node_factor

    def _family_factor(self) -> "Factor":
        # ``P(self | parents)`` as a factor over ``(*parent leaves, self)``.
        # Derived parents are folded in by evaluating them per leaf
        # assignment, so factor variables are always sources. Cost is the
        # product of the leaf supports — family-local, never joint-sized.
        assert self._fn is not None
        if self._node_factor is None:
            leaves = tuple(
                dict.fromkeys(leaf for parent in self._parents for leaf in parent._leaves())
            )
            table: "dict[tuple[Any, ...], Weight]" = {}
            for combo in product(*[leaf._support() for leaf in leaves]):
                assignment = dict(zip(leaves, combo))
                for value, weight in self._dist_for(assignment).probs.items():
                    table[(*combo, value)] = weight
            self._node_factor = Factor((*leaves, self), table)
        return self._node_factor

    def _eval(self, assignment: "Mapping[Dice[Any], Any]") -> Any:
        # The value this variable takes under a full assignment to sources.
        if self._op is None:
            return assignment[self]
        args = [
            operand._eval(assignment) if isinstance(operand, Dice) else operand
            for operand in self._operands
        ]
        return self._op(*args)

    def marginal(self) -> "Dice[Any]":
        """The marginal distribution of this variable over the model joint."""
        assert self._world is not None, "marginal() is only defined for model variables"
        return self._world.query(self)

    def believe(self, dist: "Mapping[Any, Weight] | Dice[Any]") -> None:
        """Assert a soft (Jeffrey) belief on this variable, in place.

        ``dist`` maps outcomes to weights (or is a plain ``Dice``). The
        world conditions every subsequent query on it, so the belief
        propagates across the whole graph — parents and descendants — the
        next time any variable is read (``p()``, :meth:`marginal`,
        :attr:`probs`, :meth:`World.show`). Beliefs on several variables
        are satisfied *jointly* (see :meth:`World.query`); believing again
        replaces this variable's earlier belief, and :meth:`forget`
        withdraws it. A point mass is hard evidence::

            test.believe({"positive": 1})
        """
        if self._world is None:
            raise TypeError("believe() needs a model variable; this dice is standalone")
        probs = dist.probs if isinstance(dist, Dice) else dict(dist)
        self._world._set_belief(self, probs)

    def forget(self) -> None:
        """Withdraw this variable's asserted belief (see :meth:`believe`)."""
        if self._world is None:
            raise TypeError("forget() needs a model variable; this dice is standalone")
        self._world._clear_belief(self)

    def _derive(self, op: "Callable[..., Any]", operands: "Sequence[Any]", is_predicate: bool) -> "Dice[Any]":
        # Build a derived variable from an operator. A directly-assigned
        # expression (``win = ...``) captures its name and joins the graph;
        # inner sub-expressions stay anonymous.
        name = _assigned_name()
        node = Dice._make(
            world=self._world, op=op, operands=operands, is_predicate=is_predicate, name=name
        )
        if name is not None:
            assert self._world is not None
            self._world._register_derived(node)
        return node

    def __add__(self, other: Any) -> "Dice[Any] | Self":
        if self._world is not None:
            return self._derive(operator.add, (self, other), False)
        if not isinstance(other, Dice) and other == 0:
            return self
        return self.operate(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "Dice[Any] | Self":
        if self._world is not None:
            return self._derive(operator.add, (other, self), False)
        if not isinstance(other, Dice) and other == 0:
            return self
        return self.operate(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.sub, (self, other), False)
        return self.operate(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.sub, (other, self), False)
        return self.operate(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.mul, (self, other), False)
        if isinstance(other, int) and not isinstance(other, bool):
            if other < 1:
                raise ValueError("count must be at least 1")
            return reduce(lambda a, b: a + b, [self] * other)
        return self.operate(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.mul, (other, self), False)
        if isinstance(other, int) and not isinstance(other, bool):
            return self.__mul__(other)
        return self.operate(other, lambda a, b: a * b)

    def __matmul__(self, other: Any) -> "Dice[Any]":
        return self.operate(other, lambda a, b: a * b)

    def __rmatmul__(self, other: Any) -> "Dice[Any]":
        return self.operate(other, lambda a, b: a * b)

    def __le__(self, other: Any) -> "Dice[bool]":
        if self._world is not None:
            return self._derive(operator.le, (self, other), True)
        return self.operate(other, lambda a, b: a <= b)

    def __lt__(self, other: Any) -> "Dice[bool]":
        if self._world is not None:
            return self._derive(operator.lt, (self, other), True)
        return self.operate(other, lambda a, b: a < b)

    def __ge__(self, other: Any) -> "Dice[bool]":
        if self._world is not None:
            return self._derive(operator.ge, (self, other), True)
        return self.operate(other, lambda a, b: a >= b)

    def __gt__(self, other: Any) -> "Dice[bool]":
        if self._world is not None:
            return self._derive(operator.gt, (self, other), True)
        return self.operate(other, lambda a, b: a > b)

    def __eq__(self, other: Any) -> "Dice[bool]":  # type: ignore[override]
        if self._world is not None:
            return self._derive(operator.eq, (self, other), True)
        return self.operate(other, lambda a, b: a == b)

    def __ne__(self, other: Any) -> "Dice[bool]":  # type: ignore[override]
        if self._world is not None:
            return self._derive(operator.ne, (self, other), True)
        return self.operate(other, lambda a, b: a != b)

    def __and__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.and_, (self, other), True)
        return self.operate(other, lambda a, b: a and b)

    def __or__(self, other: Any) -> "Dice[Any]":
        if self._world is not None:
            return self._derive(operator.or_, (self, other), True)
        return self.operate(other, lambda a, b: a or b)

    def __invert__(self) -> "Dice[bool]":
        if self._world is not None:
            return self._derive(operator.not_, (self,), True)
        return self.map(lambda a: not a)

    def __len__(self) -> int:
        return len(self.probs)

_active_world: "contextvars.ContextVar[World | None]" = contextvars.ContextVar(
    "dicekit_world", default=None
)


def _assigned_name() -> str | None:
    """The name in a user's ``name = <expr>`` line, read from bytecode.

    Walks outward from the caller past dicekit-internal frames (dunder
    methods like ``__init__``/``__eq__`` and the named helpers below) to
    the first user frame, then checks whether its next instruction stores
    into a name. Best-effort: returns ``None`` for inline use, tuple
    unpacking, or attribute/subscript targets.
    """
    internal = {"from_numbers", "from_sides", "from_dist", "from_function", "from_weights", "from_editor", "take", "_raw", "_make", "_derive", "_coerce_dice"}
    frame: "Any" = sys._getframe(1)
    while frame is not None and (frame.f_code.co_name.startswith("__") or frame.f_code.co_name in internal):
        frame = frame.f_back
    if frame is None:
        return None
    following = [
        instruction
        for instruction in dis.get_instructions(frame.f_code)
        if instruction.offset > frame.f_lasti
    ]
    if following and following[0].opname in {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}:
        return cast(str, following[0].argval)
    return None


def _coerce_dice(value: object) -> "Dice[Any]":
    if isinstance(value, Dice):
        return value
    return Dice._raw(cast(Mapping[Any, Weight], value))


def _jeffrey_condition(
    rows: "list[tuple[dict[Dice[Any], Any], Weight]]",
    beliefs: "Mapping[Dice[Any], Mapping[Any, Weight]]",
    max_rounds: int = 100,
    tol: float = 1e-9,
) -> "list[tuple[dict[Dice[Any], Any], Weight]]":
    """Reweight joint atoms so each believed variable's marginal matches its
    target distribution — all of them at once.

    One belief is plain Jeffrey conditioning. A single sequential pass over
    several beliefs is order-dependent (each update disturbs the earlier
    marginals when the variables are correlated), so the passes are iterated
    to their joint fixpoint (iterative proportional fitting). Target mass on
    outcomes the joint cannot reach is dropped and the remainder
    renormalized, per variable, as in single-pass Jeffrey conditioning.
    """
    if not beliefs:
        return rows
    targets: "dict[Dice[Any], dict[Any, Weight]]" = {}
    for node, dist in beliefs.items():
        scale = sum(dist.values())
        if scale <= 0:
            raise ValueError("belief must have positive total weight")
        targets[node] = {value: weight / scale for value, weight in dist.items()}
    for _ in range(max_rounds):
        deviation: Weight = 0
        for node, target in targets.items():
            marginal: "defaultdict[Any, Weight]" = defaultdict(int)
            for assignment, weight in rows:
                marginal[node._eval(assignment)] += weight
            total = sum(marginal.values())
            if total == 0:
                raise ValueError("evidence has zero probability")
            reachable = sum(q for value, q in target.items() if marginal[value] > 0)
            if reachable == 0:
                raise ValueError("evidence has zero probability")
            adjusted = {value: q / reachable for value, q in target.items() if marginal[value] > 0}
            deviation = max(
                deviation,
                *[abs(marginal[value] / total - adjusted.get(value, 0)) for value in {*marginal, *adjusted}],
            )
            rows = [
                (assignment, weight * adjusted.get(value, 0) * total / marginal[value])
                for assignment, weight in rows
                for value in (node._eval(assignment),)
                if marginal[value] > 0
            ]
        if deviation <= tol:
            break
    return rows


# -- the factor-graph engine -------------------------------------------
# The model's joint is a *product of small tables*: one prior factor per
# root, one family factor per dependent. Inference multiplies and sums
# these out (variable elimination) instead of enumerating the joint, so
# cost grows with treewidth — how tangled the graph is — not with the
# number of variables. The exhaustive enumeration survives as
# ``World.joint()``: the reference oracle for tests and the data behind
# small-model teaching views. Messages for sum-product propagation (and a
# loopy schedule for high-treewidth models) can be layered on the same
# Factor type later without rework.

class Factor:
    """A sparse weight table over a tuple of source variables.

    Missing rows are zero, which keeps deterministic factors small, and
    weights keep their exact types (Fractions in → Fractions out).
    Variables are model-bound ``Dice``; because their ``==`` builds an
    expression, all bookkeeping here is by identity (id-keyed), never by
    list membership.
    """

    __slots__ = ("variables", "table")

    def __init__(self, variables: "Sequence[Dice[Any]]", table: "Mapping[tuple[Any, ...], Weight]") -> None:
        self.variables = tuple(variables)
        self.table = dict(table)

    def __mul__(self, other: "Factor") -> "Factor":
        mine = {id(v) for v in self.variables}
        shared = tuple(i for i, v in enumerate(other.variables) if id(v) in mine)
        extra = tuple(i for i, v in enumerate(other.variables) if id(v) not in mine)
        position = {id(v): i for i, v in enumerate(self.variables)}
        shared_mine = tuple(position[id(other.variables[i])] for i in shared)

        buckets: "defaultdict[tuple[Any, ...], list[tuple[tuple[Any, ...], Weight]]]" = defaultdict(list)
        for key, weight in other.table.items():
            buckets[tuple(key[i] for i in shared)].append((tuple(key[i] for i in extra), weight))

        joined: "dict[tuple[Any, ...], Weight]" = {}
        for key, weight in self.table.items():
            match = tuple(key[i] for i in shared_mine)
            for extra_key, other_weight in buckets.get(match, ()):
                row = key + extra_key
                joined[row] = joined.get(row, 0) + weight * other_weight
        return Factor(self.variables + tuple(other.variables[i] for i in extra), joined)

    def sum_out(self, variable: "Dice[Any]") -> "Factor":
        keep = [i for i, v in enumerate(self.variables) if v is not variable]
        summed: "defaultdict[tuple[Any, ...], Weight]" = defaultdict(int)
        for key, weight in self.table.items():
            summed[tuple(key[i] for i in keep)] += weight
        return Factor(tuple(self.variables[i] for i in keep), dict(summed))


def _eliminate(factors: "Sequence[Factor]", keep: "Sequence[Dice[Any]]") -> Factor:
    """Multiply-and-sum-out every variable not in ``keep``.

    The next variable is chosen greedily by *min-fill* (fewest edges its
    elimination would add between its neighbors), the classic heuristic
    that keeps intermediate tables near the treewidth bound. Variables
    whose neighbors are already a clique — in particular leaves — are
    picked immediately.
    """
    keep_ids = {id(v) for v in keep}
    factors = list(factors)
    while True:
        variables: "dict[int, Dice[Any]]" = {}
        neighbors: "defaultdict[int, set[int]]" = defaultdict(set)
        for factor in factors:
            for v in factor.variables:
                variables[id(v)] = v
                for u in factor.variables:
                    if u is not v:
                        neighbors[id(v)].add(id(u))
        candidates = [vid for vid in variables if vid not in keep_ids]
        if not candidates:
            break

        def fill(vid: int) -> "tuple[int, int]":
            around = [n for n in neighbors[vid] if n in variables]
            missing = sum(
                1
                for i, a in enumerate(around)
                for b in around[i + 1:]
                if b not in neighbors[a]
            )
            return (missing, len(around))

        best = None
        best_fill = None
        for vid in sorted(candidates, key=lambda c: len(neighbors[c])):
            if len(neighbors[vid]) <= 1:
                best = vid  # a (near-)leaf never adds edges
                break
            score = fill(vid)
            if best_fill is None or score < best_fill:
                best, best_fill = vid, score
                if score[0] == 0:
                    break
        chosen = variables[cast(int, best)]

        bucket = [f for f in factors if any(v is chosen for v in f.variables)]
        rest = [f for f in factors if not any(v is chosen for v in f.variables)]
        merged = bucket[0]
        for factor in bucket[1:]:
            merged = merged * factor
        factors = rest + [merged.sum_out(chosen)]

    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


def _indicator_factor(expr: "Dice[Any]", weight_fn: "Callable[[Any], Weight]") -> Factor:
    """A factor over ``expr``'s source leaves weighing each assignment by
    ``weight_fn(expr's value there)`` — how hard evidence and beliefs on
    derived expressions enter the factor product. Cost is the product of
    the leaf supports, i.e. genuinely wide expressions cost real treewidth.
    """
    leaves = expr._leaves()
    table: "dict[tuple[Any, ...], Weight]" = {}
    for combo in product(*[leaf._support() for leaf in leaves]):
        weight = weight_fn(expr._eval(dict(zip(leaves, combo))))
        if weight:
            table[combo] = weight
    return Factor(leaves, table)


def _marginal_of(factors: "Sequence[Factor]", node: "Dice[Any]") -> "dict[Any, Weight]":
    """The (unnormalized) marginal of ``node`` under a factor product.

    Sources marginalize directly; a derived expression is marginalized
    onto its leaf set and then evaluated — it never becomes a column.
    """
    leaves = node._leaves()
    joint = _eliminate(factors, leaves)
    index = {id(v): i for i, v in enumerate(joint.variables)}
    weights: "defaultdict[Any, Weight]" = defaultdict(int)
    for key, weight in joint.table.items():
        assignment = {leaf: key[index[id(leaf)]] for leaf in leaves}
        weights[node._eval(assignment)] += weight
    return dict(weights)


def _sum_product(
    factors: "Sequence[Factor]",
) -> "tuple[dict[int, dict[Any, Weight]], dict[tuple[Any, Any], dict[Any, Weight]]] | None":
    """Two-pass sum-product over the factor graph, if it is a forest.

    Returns ``(marginals, messages)``: unnormalized marginals for every
    source variable (keyed by ``id``) from ONE calibration — the fast path
    for "repaint every node" — and every edge message, the objects a
    message-passing visualization draws. Nodes are ``("v", id(var))`` and
    ``("f", index)``; a message key is ``(src, dst)``. Returns ``None``
    when the factor graph has a cycle; callers fall back to variable
    elimination (exact at treewidth cost). A loopy schedule — the same
    messages iterated with damping — can reuse this structure later.
    """
    variables: "dict[int, Dice[Any]]" = {}
    adjacency: "defaultdict[Any, list[Any]]" = defaultdict(list)
    union: "dict[Any, Any]" = {}

    def find(node: Any) -> Any:
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node

    for index, factor in enumerate(factors):
        fnode = ("f", index)
        union.setdefault(fnode, fnode)
        adjacency[fnode]  # ensure isolated (constant) factors exist as nodes
        for v in factor.variables:
            vnode = ("v", id(v))
            variables[id(v)] = v
            union.setdefault(vnode, vnode)
            root_f, root_v = find(fnode), find(vnode)
            if root_f == root_v:
                return None  # a cycle: this graph is loopy
            union[root_f] = root_v
            adjacency[fnode].append(vnode)
            adjacency[vnode].append(fnode)

    messages: "dict[tuple[Any, Any], dict[Any, Weight]]" = {}

    def compute(src: Any, dst: Any) -> "dict[Any, Weight]":
        if src[0] == "v":
            v = variables[src[1]]
            out = {value: cast(Weight, 1) for value in v._support()}
            for neighbor in adjacency[src]:
                if neighbor == dst:
                    continue
                incoming = messages[(neighbor, src)]
                out = {value: weight * incoming.get(value, 0) for value, weight in out.items()}
        else:
            factor = factors[src[1]]
            position = next(i for i, u in enumerate(factor.variables) if id(u) == dst[1])
            incoming_by_index = [
                None if i == position else messages.get((("v", id(u)), src))
                for i, u in enumerate(factor.variables)
            ]
            summed: "defaultdict[Any, Weight]" = defaultdict(int)
            for key, weight in factor.table.items():
                for i, incoming in enumerate(incoming_by_index):
                    if incoming is not None:
                        weight = weight * incoming.get(key[i], 0)
                        if weight == 0:
                            break
                summed[key[position]] += weight
            out = dict(summed)
        total = sum(out.values())
        if total > 0:  # normalize so long chains cannot underflow
            out = {value: weight / total for value, weight in out.items()}
        return out

    # strip leaves inward (collect), then unwind outward (distribute)
    remaining = {node: len(neighbors) for node, neighbors in adjacency.items()}
    stripped: set[Any] = set()
    parent_of: "dict[Any, Any]" = {}
    order: "list[Any]" = []
    frontier = [node for node, count in remaining.items() if count <= 1]
    while frontier:
        node = frontier.pop()
        if node in stripped:
            continue
        alive = [n for n in adjacency[node] if n not in stripped]
        parent = alive[0] if alive else None
        parent_of[node] = parent
        stripped.add(node)
        order.append(node)
        if parent is not None:
            messages[(node, parent)] = compute(node, parent)
            remaining[parent] -= 1
            if remaining[parent] <= 1:
                frontier.append(parent)
    for node in reversed(order):
        parent = parent_of[node]
        if parent is not None:
            messages[(parent, node)] = compute(parent, node)

    marginals: "dict[int, dict[Any, Weight]]" = {}
    for vid, v in variables.items():
        out = {value: cast(Weight, 1) for value in v._support()}
        for neighbor in adjacency[("v", vid)]:
            incoming = messages[(neighbor, ("v", vid))]
            out = {value: weight * incoming.get(value, 0) for value, weight in out.items()}
        marginals[vid] = out
    return marginals, messages


def _belief_marginal(factors: "Sequence[Factor]", node: "Dice[Any]") -> "dict[Any, Weight]":
    # A single node's marginal, via one cheap calibration when the graph
    # is a forest and the node is a source; elimination otherwise.
    if node._op is None:
        calibrated = _sum_product(factors)
        if calibrated is not None:
            return dict(calibrated[0].get(id(node), {}))
    return _marginal_of(factors, node)


def _ipf_factors(
    factors: "Sequence[Factor]",
    beliefs: "Mapping[Dice[Any], Mapping[Any, Weight]]",
    max_rounds: int = 100,
    tol: float = 1e-9,
) -> "list[Factor]":
    """Belief adjustments as per-variable factors, fitted by IPF.

    The factor-engine twin of :func:`_jeffrey_condition`: instead of
    reweighting joint atoms, each believed variable gets a unary (or, for
    a derived expression, leaf-indicator) adjustment factor, iterated to
    the joint fixpoint — same semantics, but each pass costs an
    elimination, not a joint scan. Target mass on unreachable outcomes is
    dropped and the rest renormalized, per variable, as before.
    """
    if not beliefs:
        return []
    targets: "dict[Dice[Any], dict[Any, Weight]]" = {}
    for node, dist in beliefs.items():
        scale = sum(dist.values())
        if scale <= 0:
            raise ValueError("belief must have positive total weight")
        targets[node] = {value: weight / scale for value, weight in dist.items()}
    ratios: "dict[Dice[Any], dict[Any, Weight]]" = {
        node: {value: 1 for value in node._support()} for node in targets
    }

    def adjustments() -> "list[Factor]":
        built = []
        for node, ratio in ratios.items():
            if node._op is None:
                built.append(Factor((node,), {(v,): r for v, r in ratio.items() if r}))
            else:
                built.append(_indicator_factor(node, lambda v, ratio=ratio: ratio.get(v, 0)))
        return built

    base = list(factors)
    for _ in range(max_rounds):
        deviation: Weight = 0
        for node, target in targets.items():
            marginal = _belief_marginal(base + adjustments(), node)
            total = sum(marginal.values())
            if total == 0:
                raise ValueError("evidence has zero probability")
            reachable = {v: q for v, q in target.items() if marginal.get(v, 0) > 0}
            reachable_total = sum(reachable.values())
            if reachable_total == 0:
                raise ValueError("evidence has zero probability")
            adjusted = {v: q / reachable_total for v, q in reachable.items()}
            deviation = max(
                deviation,
                *[abs(marginal.get(v, 0) / total - adjusted.get(v, 0)) for v in {*marginal, *adjusted}],
            )
            ratio = ratios[node]
            for v in list(ratio):
                mass = marginal.get(v, 0)
                if mass > 0:
                    ratio[v] = ratio[v] * adjusted.get(v, 0) * total / mass
        if deviation <= tol:
            break
    return adjustments()


def _belief_plot(seed: "Dice[Any]", touched: bool = False) -> Any:
    """The interactive node chart for :meth:`World.show`.

    A posterior bar chart with an ``edit`` toggle: while editing, dragging
    the bars asserts the dragged distribution as a belief (``touched`` in
    the synced ``variables``) and ``reset`` withdraws it. The kernel side
    (:meth:`World._wire_belief_editors`) listens on ``variables`` and keeps
    every other node's chart in sync.
    """
    data = [{"i": _plot_value(k), "p": _plot_value(v)} for k, v in seed.probs.items()]
    return ObservablePlot(
        _EDITOR_JS,
        variables={
            "data": data,
            "editMode": "locked",
            "uiChrome": "belief",
            "faces": [row["i"] for row in data],
            "weights": [row["p"] for row in data],
            "touched": touched,
            "editing": False,
        },
        width=240,
        height=190,
    )


def _widget_touched(widget: Any) -> bool:
    """Whether an editor widget has actually been dragged.

    A fresh editor still holds its seeded distribution; treating it as
    evidence would *assert the seed* (e.g. re-clamp a variable to its
    prior), so untouched widgets are ignored by :meth:`World.show`.
    """
    variables = getattr(widget, "variables", None)
    if variables is None:
        value = getattr(widget, "value", {}) or {}
        variables = value.get("variables", value)
    return bool(variables.get("touched"))


class World:
    """A shared exact joint over the variables created within it.

    Yielded by ``with model() as m:``. Tracks the model's source variables
    (roots + dependents, in creation order so a child's parents are bound
    before it) and any named derived expressions, and answers marginal /
    conditional queries **by variable elimination over the model's
    factors** — cost grows with treewidth (how tangled the graph is), not
    with the number of variables, so tree-ish models scale to hundreds of
    variables. The exhaustive enumeration survives as :meth:`joint`, the
    reference oracle for tests and small-model teaching views.

    Also carries the model's mutable *beliefs*: soft evidence asserted via
    :meth:`Dice.believe` (or by dragging a node in :meth:`show`). Every
    query conditions on them, so an updated belief propagates across the
    whole graph on the next read; :meth:`forget` clears them all.
    """

    def __init__(self) -> None:
        self.sources: "list[Dice[Any]]" = []
        self.derived: "list[Dice[Any]]" = []
        self._factor_list: "list[Factor] | None" = None
        self._factor_count = 0
        # Asserted beliefs (soft evidence) per variable — the world's
        # mutable state. Every query conditions on them; the version
        # counter invalidates the lazy `probs` caches, and listeners let
        # rendered graphs repaint in place.
        self._beliefs: "dict[Dice[Any], dict[Any, Weight]]" = {}
        self._belief_version = 0
        self._belief_listeners: "list[Callable[[Dice[Any] | None], None]]" = []

    def _register(self, variable: "Dice[Any]") -> None:
        self.sources.append(variable)

    def _register_derived(self, variable: "Dice[Any]") -> None:
        self.derived.append(variable)

    def _set_belief(self, node: "Dice[Any]", dist: "Mapping[Any, Weight]") -> None:
        if node._world is not self:
            raise ValueError("belief must be about a variable of this model")
        probs = dict(dist)
        if not probs or sum(probs.values()) <= 0:
            raise ValueError("belief must have positive total weight")
        self._beliefs[node] = probs
        self._belief_changed(node)

    def _clear_belief(self, node: "Dice[Any]") -> None:
        if self._beliefs.pop(node, None) is not None:
            self._belief_changed(node)

    def forget(self) -> None:
        """Withdraw every asserted belief (see :meth:`Dice.believe`)."""
        if self._beliefs:
            self._beliefs.clear()
            self._belief_changed(None)

    def _display_(self) -> Any:
        # a bare `m` in a marimo cell renders the interactive graph
        return self.show()

    def _belief_changed(self, node: "Dice[Any] | None") -> None:
        self._belief_version += 1
        for listener in list(self._belief_listeners):
            listener(node)

    def joint(self) -> "list[tuple[dict[Dice[Any], Any], Weight]]":
        """The exhaustive weighted joint, one row per full assignment.

        Exponential in the number of variables — kept as the reference
        implementation (tests cross-check the factor engine against it)
        and as the data behind small-model views like the atoms chart.
        Inference itself goes through :meth:`_factors` + elimination.
        """
        rows: "list[tuple[dict[Dice[Any], Any], Weight]]" = [({}, 1)]
        for source in self.sources:
            expanded: "list[tuple[dict[Dice[Any], Any], Weight]]" = []
            for assignment, weight in rows:
                for value, probability in source._dist_for(assignment).probs.items():
                    expanded.append(({**assignment, source: value}, weight * probability))
            rows = expanded
        return rows

    def _factors(self) -> "list[Factor]":
        # One factor per source: priors for roots, families for dependents.
        # Cached; rebuilt only if more variables were registered.
        if self._factor_list is None or self._factor_count != len(self.sources):
            token = _active_world.set(None)
            try:
                self._factor_list = [
                    source._family_factor() if source._fn is not None else source._prior_factor()
                    for source in self.sources
                ]
            finally:
                _active_world.reset(token)
            self._factor_count = len(self.sources)
        return list(self._factor_list)

    def _posteriors(
        self,
        given: "Dice[Any] | None" = None,
        evidence: "Mapping[Dice[Any], Mapping[Any, Weight]] | None" = None,
    ) -> "dict[int, Dice[Any]]":
        """Posterior marginals for every source and named derived variable,
        keyed by ``id`` — what a full graph repaint needs. One sum-product
        calibration covers all sources when the factor graph is a forest
        (and stashes its messages on ``_last_messages``); loopy graphs and
        derived expressions fall back to per-node elimination.
        """
        token = _active_world.set(None)
        try:
            factors = self._factors()
            if given is not None:
                factors.append(_indicator_factor(given, lambda value: 1 if value else 0))
            factors += _ipf_factors(factors, {**self._beliefs, **(evidence or {})})
            calibrated = _sum_product(factors)
            marginals = calibrated[0] if calibrated is not None else None
            self._last_messages = calibrated[1] if calibrated is not None else None
            result: "dict[int, Dice[Any]]" = {}
            for node in [*self.sources, *self.derived]:
                if marginals is not None and node._op is None:
                    weights = marginals.get(id(node), {})
                else:
                    weights = _marginal_of(factors, node)
                if sum(weights.values()) == 0:
                    raise ValueError("evidence has zero probability")
                order = {value: index for index, value in enumerate(node._support())}
                fallback = len(order)
                result[id(node)] = Dice._raw(
                    {value: weights[value] for value in sorted(weights, key=lambda value: order.get(value, fallback))}
                )
            return result
        finally:
            _active_world.reset(token)

    def query(
        self,
        target: "Dice[Any]",
        given: "Dice[Any] | None" = None,
        evidence: "Mapping[Dice[Any], Mapping[Any, Weight]] | None" = None,
    ) -> "Dice[Any]":
        """Marginal of ``target``, optionally under evidence.

        ``given`` is *hard* evidence: a boolean predicate that must hold.

        ``evidence`` is *soft* (Jeffrey) evidence: a mapping from a variable
        to a target distribution over its outcomes. The joint is reweighted so
        each listed variable's posterior marginal becomes that distribution,
        and the belief propagates to every other variable — descendants *and*
        parents, exactly as a probabilistic graphical model dictates. Hard
        evidence is the point-mass special case. Several soft-evidence
        variables are satisfied *jointly* (iterative proportional fitting),
        so the result does not depend on their order.

        Beliefs asserted on the world (:meth:`Dice.believe`) are always
        folded in; ``evidence`` overrides a belief on the same variable.

        Runs on the factor engine: variable elimination over the model's
        factors, plus an indicator factor for ``given`` and IPF-fitted
        adjustment factors for the beliefs — exact, with cost bounded by
        treewidth rather than the joint size.
        """
        # Work with the model deactivated, so dice built by conditional
        # functions (and the result itself) are not registered as variables.
        token = _active_world.set(None)
        try:
            factors = self._factors()
            if given is not None:
                factors.append(_indicator_factor(given, lambda value: 1 if value else 0))
            factors += _ipf_factors(factors, {**self._beliefs, **(evidence or {})})
            weights = _marginal_of(factors, target)
            if sum(weights.values()) == 0:
                raise ValueError("evidence has zero probability")
            # canonical outcome order (the support's), so chart axes stay
            # stable instead of following elimination order
            order = {value: index for index, value in enumerate(target._support())}
            fallback = len(order)
            weights = {
                value: weights[value]
                for value in sorted(weights, key=lambda value: order.get(value, fallback))
            }
            return Dice._raw(weights)
        finally:
            _active_world.reset(token)

    def _graph_parents(self, node: "Dice[Any]", tracked_ids: "set[int]") -> "list[Dice[Any]]":
        # The nearest tracked variables a derived node is built from, so it
        # connects to what it actually depends on, skipping anonymous
        # intermediates. id()-keyed because `==` builds an expression.
        found: "list[Dice[Any]]" = []
        found_ids: set[int] = set()

        def walk(current: "Dice[Any]") -> None:
            for operand in current._operands:
                if not isinstance(operand, Dice):
                    continue
                if id(operand) in tracked_ids:
                    if id(operand) not in found_ids:
                        found_ids.add(id(operand))
                        found.append(operand)
                else:
                    walk(operand)

        walk(node)
        return found

    def _belief_entries(
        self,
        given: "Dice[Any] | None" = None,
        soft: "Mapping[Dice[Any], Mapping[Any, Weight]] | None" = None,
        skip_ids: "set[int] | frozenset[int]" = frozenset(),
        nodes: "Sequence[Dice[Any]] | None" = None,
    ) -> "list[tuple[Dice[Any], Any, list[Any]]]":
        """Build one interactive belief chart per variable and wire it up.

        Returns ``(node, plot, outcomes)`` triples; the outcomes list fixes
        the face order so dragged weights can be mapped back to the node's
        real outcome objects (the widget's faces round-trip through JSON).
        """
        soft = dict(soft or {})
        posteriors = self._posteriors(given, soft)
        entries: "list[tuple[Dice[Any], Any, list[Any]]]" = []
        for node in [*self.sources, *self.derived] if nodes is None else list(nodes):
            if id(node) in skip_ids:
                continue
            posterior = posteriors[id(node)]
            plot = _belief_plot(posterior, touched=node in self._beliefs)
            entries.append((node, plot, list(posterior.probs)))
        self._wire_belief_editors(entries, given, soft)
        return entries

    def _wire_belief_editors(
        self,
        entries: "list[tuple[Dice[Any], Any, list[Any]]]",
        given: "Dice[Any] | None",
        soft: "Mapping[Dice[Any], Mapping[Any, Weight]]",
    ) -> None:
        # The in-widget propagation loop: a drag commit updates the plot's
        # `variables` trait kernel-side, the observer asserts it as a
        # belief, and the belief listener pushes fresh posteriors into
        # every other chart — no notebook re-runs involved, so it works
        # identically in marimo and Jupyter.
        if not entries:
            return
        pushing = {"active": False}

        def refresh(origin: "Dice[Any] | None") -> None:
            if pushing["active"]:
                return
            pushing["active"] = True
            try:
                # one calibration for the whole graph, not a query per node
                posteriors = self._posteriors(given, soft)
                for node, plot, outcomes in entries:
                    if node is origin and node in self._beliefs:
                        continue  # don't repaint the editor being dragged
                    try:
                        posterior = posteriors[id(node)]
                        weights = [_plot_value(posterior.probs.get(outcome, 0)) for outcome in outcomes]
                        faces = [_plot_value(outcome) for outcome in outcomes]
                        plot.variables = {
                            **plot.variables,
                            "data": [{"i": face, "p": weight} for face, weight in zip(faces, weights)],
                            "weights": weights,
                            "faces": faces,
                            "touched": node in self._beliefs,
                        }
                    except Exception:
                        continue  # a stale widget (closed comm) must not break the rest
            except Exception:
                pass  # e.g. transiently impossible evidence mid-drag
            finally:
                pushing["active"] = False

        def on_change(change: Any, node: "Dice[Any]", outcomes: "list[Any]") -> None:
            if pushing["active"]:
                return
            variables = change["new"] or {}
            weights = variables.get("weights") or []
            if variables.get("touched") and len(weights) == len(outcomes) and sum(weights) > 0:
                # zip against the *original* outcomes: JSON round-trips the faces
                node.believe(dict(zip(outcomes, weights)))
            elif not variables.get("touched") and node in self._beliefs:
                node.forget()

        for node, plot, outcomes in entries:
            plot.observe(
                lambda change, node=node, outcomes=outcomes: on_change(change, node, outcomes),
                names="variables",
            )
        self._belief_listeners.append(refresh)

    def show(
        self,
        given: "Dice[Any] | None" = None,
        inputs: "Mapping[Dice[Any], Any] | None" = None,
        editable: "Mapping[Dice[Any], Any] | None" = None,
        evidence: "Mapping[Dice[Any], Any] | None" = None,
    ) -> Any:
        """Render the model as a DAG of live, editable belief charts (marimo only).

        Each source and each named derived variable becomes a node,
        labelled with its Python name, with an arrow for every dependency,
        via wigglystuff's ``WidgetDAG``.

        Every node is interactive by default, with two states: *live*
        (black — it flows with the model and repaints as beliefs change)
        and *editing* (blue — click ``edit``, then drag its bars to assert
        a soft belief on that variable, :meth:`Dice.believe`). Every other
        node's posterior repaints in place, parents included. ``reset`` is
        the one way back: it withdraws the belief (:meth:`Dice.forget`)
        and returns the node to the live flow. The update loop runs
        through the widgets' synced state, not notebook re-runs, so
        ``m.show()`` is the whole line: no editor cells, no evidence
        dicts. Programmatic :meth:`Dice.believe` calls repaint the graph
        too, and beliefs asserted by dragging are visible to ``p()`` /
        :meth:`query` in other cells (marimo does not re-run those cells
        on a drag — re-run to read the new posterior).

        ``given`` conditions the charts: pass a predicate (e.g.
        ``test == "positive"``) and every node shows its *posterior*
        marginal rather than the prior, so evidence visibly propagates.

        ``inputs`` maps a target variable to a widget (e.g. a slider), drawn
        as an extra upstream node with an arrow into its target; read the
        widget's ``.value`` inside the ``variable`` it feeds for reactivity.

        ``editable`` maps a *source* variable to a :meth:`editor` widget,
        rendering that node *as the editor itself* so you can drag its prior
        right inside the graph. Build the model's prior for that variable from
        the same widget via :meth:`from_editor`; because marimo re-runs the
        model cell on every drag, the whole graph recomputes live. This edits
        the *generative model*.

        ``evidence`` maps *any* variable to an externally-created
        :meth:`editor` widget and treats the dragged distribution as soft
        (Jeffrey) *evidence* on that node, via marimo reactivity. An
        editor that has not been dragged yet is ignored — treating its
        seed as evidence would clamp the variable to it. With the built-in
        ``edit`` toggle this wiring is rarely needed anymore.
        """
        import marimo as mo
        from wigglystuff import WidgetDAG

        graph_nodes = [*self.sources, *self.derived]
        tracked_ids = {id(node) for node in graph_nodes}
        # display names, made unique so same-named variables (e.g. built in
        # a loop) don't silently collapse into one WidgetDAG node
        names: dict[int, str] = {}
        used_names: set[str] = set()
        for index, node in enumerate(graph_nodes):
            name = node._name or f"var{index}"
            while name in used_names:
                name += "'"
            used_names.add(name)
            names[id(node)] = name

        editable_by_id = {id(node): widget for node, widget in (editable or {}).items()}
        evidence_by_id = {id(node): widget for node, widget in (evidence or {}).items()}
        widget_by_id = {**editable_by_id, **evidence_by_id}
        # *dragged* evidence editors become soft evidence for querying the
        # rest; untouched ones are inert rather than asserting their seed
        soft = {
            node: Dice.from_editor(widget).probs
            for node, widget in (evidence or {}).items()
            if _widget_touched(widget)
        }

        entries = self._belief_entries(given, soft, skip_ids=set(widget_by_id))
        nodes: dict[str, Any] = {}
        interactive = {id(node): plot for node, plot, _ in entries}
        for node in graph_nodes:
            if id(node) in widget_by_id:
                nodes[names[id(node)]] = widget_by_id[id(node)]
            else:
                nodes[names[id(node)]] = mo.ui.anywidget(interactive[id(node)])

        edges: list[tuple[str, str]] = []
        for source in self.sources:
            edges.extend((names[id(parent)], names[id(source)]) for parent in source._parents)
        for node in self.derived:
            edges.extend(
                (names[id(parent)], names[id(node)])
                for parent in self._graph_parents(node, tracked_ids)
            )

        for target, widget in (inputs or {}).items():
            node_id = f"{names.get(id(target), 'input')} input"
            while node_id in nodes:
                node_id += "'"
            nodes[node_id] = widget
            if id(target) in names:
                edges.append((node_id, names[id(target)]))
        return WidgetDAG(nodes, edges)


@contextmanager
def model() -> "Iterator[World]":
    """Open a fresh model. Dice created inside the block share one joint.

    Because a ``with`` block introduces no new scope, the variables stay
    bound as ordinary names afterwards (and, in marimo, as top-level cell
    definitions that plot).
    """
    world = World()
    token = _active_world.set(world)
    try:
        yield world
    finally:
        _active_world.reset(token)

class Vase:
    """
    A collection of items that can be drawn to form a probability distribution.

    A vase preserves repeated items and can model draws with or without
    replacement, where the order of drawn items may optionally matter.
    """

    def __init__(self, contents: Sequence[str]) -> None:
        """
        Initialize a vase with the items it contains.

        Parameters:
            contents (list): Items available to draw from the vase
        """
        self._contents = contents

    @classmethod
    def from_counts(cls, **kwargs: int) -> "Vase":
        """
        Create a vase from item names and their quantities.

        Parameters:
            **kwargs (int): Item names mapped to the number of copies

        Returns:
            Vase: A vase containing the requested number of each item
        """
        contents = []
        for k, v in kwargs.items():
            contents.extend([k]*v)
        return cls(contents)

    def _to_sorted_key(self, tup: Sequence[str]) -> str:
        """
        Convert a collection of items into an order-independent key.

        Parameters:
            tup (tuple): Items drawn from the vase

        Returns:
            str: The items sorted and joined into a single key
        """
        return "".join(sorted(tup))

    def take(self, n: int = 1, replace: bool = False, ordered: bool = False) -> "Dice[str]":
        """
        Calculate the distribution of drawing items from the vase.

        Parameters:
            n (int): Number of items to draw, default is 1
            replace (bool): Whether drawn items are replaced, default is False
            ordered (bool): Whether draw order affects outcomes, default is False

        Returns:
            Dice: The probability distribution over possible draws
        """
        if replace:
            out = product(self._contents, repeat=n)
        else:
            out = permutations(self._contents, n)
        out = [self._to_sorted_key(_) if not ordered else "".join(_) for _ in out]
        return Dice(Counter(out))

from functools import singledispatch


@singledispatch
def p(expression: object, given: object = None) -> Any:
    """
    Evaluate a probability expression.

    For a boolean Dice this returns P(True). The DAG layer in
    ``dicekit.learn`` registers handlers for its node and event types,
    so importing that module teaches ``p`` to query a DAG.
    """
    raise TypeError("p expects a Dice, DAG node, or DAG event")


@p.register(Dice)
def _p_dice(expression: "Dice[Any]", given: object = None) -> Any:
    world = expression._world
    if world is None:
        # A standalone (boolean) dice: probability of True.
        return expression.probs.get(True, 0)
    # A model variable: query the joint, optionally conditioning on a
    # predicate from the same model. A predicate variable collapses to the
    # scalar P(True); any other variable returns its (posterior) marginal.
    if given is not None and not (isinstance(given, Dice) and given._world is world):
        raise TypeError("given must be a predicate variable from the same model")
    dice = world.query(expression, cast("Dice[Any] | None", given))
    if expression._is_predicate:
        return dice.probs.get(True, 0)
    return dice


def exp(dice: "Dice[Weight]") -> Weight:
    """
    Calculates the expected value (mean) of a dice.

    Parameters:
        dice: A Dice object

    Returns:
        The expected value of the dice (a Fraction when the dice is exact)
    """
    return sum(i * p for i, p in dice.probs.items())


def var(dice: "Dice[Weight]") -> Weight:
    """
    Calculates the variance of a dice.

    Parameters:
        dice: A Dice object

    Returns:
        The variance of the dice (a Fraction when the dice is exact)
    """
    return sum(p * (i - exp(dice)) ** 2 for i, p in dice.probs.items())

def mix[T: Hashable](*dice: "Dice[T]", weights: Sequence[Weight] | None = None) -> "Dice[T]":
    """
    Create a weighted mixture of dice.

    Parameters:
        *dice: Dice objects to include in the mixture
        weights: Optional non-negative weight for each dice

    Returns:
        Dice: A dice representing the weighted mixture
    """
    if not dice:
        raise ValueError("mix requires at least one dice")
    if not all(isinstance(die, Dice) for die in dice):
        raise TypeError("mix expects Dice objects")

    if weights is None:
        weights = [1] * len(dice)
    if len(weights) != len(dice):
        raise ValueError("weights must contain one value for each dice")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("weights must have a positive sum")

    new_probs: dict[T, Weight] = {}
    for die, weight in zip(dice, weights):
        for outcome, probability in die.probs.items():
            contribution = probability * weight / total_weight
            new_probs[outcome] = new_probs.get(outcome, 0) + contribution
    return Dice(new_probs)


def _slice_limit(length: int, stop: int) -> int:
    return len(range(length)[:stop])


def _threshold_probabilities[T: Hashable](die: "Dice[T]", thresholds: Sequence[T]) -> dict[T, Weight]:
    """
    Return P(die >= threshold) for each threshold.
    """
    probabilities: dict[T, Weight] = {}
    cumulative: Weight = 0
    items = sorted(cast(list[Any], list(die.probs.items())), reverse=True)
    item_index = 0

    for threshold in thresholds:
        while item_index < len(items) and items[item_index][0] >= cast(Any, threshold):
            cumulative += items[item_index][1]
            item_index += 1
        probabilities[threshold] = cumulative

    return probabilities


def ordered[T: Hashable](*dice_in: "Dice[T]", k: int | None = None) -> list["Dice[T]"]:
    """
    Return dice that represent order statistics. Highest first.

    Takes multiple dice objects and returns new dice objects that represent
    the ordered outcomes (e.g., highest roll, second highest roll, etc.).

    Parameters:
        *dice_in: Variable number of Dice objects

    Returns:
        list: A list of Dice objects representing order statistics
    """
    if not dice_in:
        return []

    n_dice = len(dice_in)
    n_outputs = n_dice if k is None else _slice_limit(n_dice, k)
    if n_outputs == 0:
        return []

    thresholds = sorted(cast(list[Any], list({outcome for die in dice_in for outcome in die.probs})), reverse=True)
    probabilities = [
        _threshold_probabilities(die, thresholds)
        for die in dice_in
    ]

    # Seed with int 0/1 (not 0.0/1.0) so exact Fraction probabilities stay exact;
    # the float path is unaffected (int * float == float * float).
    zero: Weight = 0
    one: Weight = 1

    dice_out: list[dict[T, Weight]] = [{} for _ in range(n_outputs)]
    higher_threshold_survival: list[Weight] = [zero] * (n_outputs + 1)

    for threshold in thresholds:
        pass_count_probs: list[Weight] = [zero] * (n_outputs + 1)
        pass_count_probs[0] = one

        for probability in probabilities:
            p_ge = probability[threshold]
            p_lt = 1 - p_ge
            new_pass_count_probs: list[Weight] = [zero] * (n_outputs + 1)

            for pass_count, pass_count_probability in enumerate(pass_count_probs):
                if pass_count_probability == 0:
                    continue
                new_pass_count_probs[pass_count] += pass_count_probability * p_lt
                capped_pass_count = min(pass_count + 1, n_outputs)
                new_pass_count_probs[capped_pass_count] += (
                    pass_count_probability * p_ge
                )

            pass_count_probs = new_pass_count_probs

        survival: list[Weight] = [zero] * (n_outputs + 1)
        running: Weight = 0
        for pass_count in range(n_outputs, -1, -1):
            running += pass_count_probs[pass_count]
            survival[pass_count] = running

        for order_index in range(n_outputs):
            order = order_index + 1
            probability = survival[order] - higher_threshold_survival[order]
            if probability < 0 and abs(probability) < 1e-15:
                probability = 0
            if probability != 0:
                dice_out[order_index][threshold] = probability

        higher_threshold_survival = survival

    return [Dice(probs) for probs in dice_out]
