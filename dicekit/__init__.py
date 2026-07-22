__all__ = ['Weight', 'Dice', 'World', 'model', 'Vase', 'p', 'exp', 'var', 'mix', 'ordered']


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
        computed once as the marginal over the model's joint, then cached.
        """
        if self._probs is None:
            assert self._world is not None
            self._probs = self._world.query(self)._probs
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

    def _repr_mimebundle_(self, **kwargs: Any) -> Any:
        """
        Render the dice as a live Observable Plot widget in notebooks.

        Delegates to :meth:`prob_chart`, so a bare ``Dice`` displays as its
        probability mass function (an ``ObservablePlot`` anywidget) rather
        than static HTML.
        """
        return self.prob_chart()._repr_mimebundle_(**kwargs)

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
            """
            const n = data.length;
            const mode = typeof editMode === "string" ? editMode : "renormalize";
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
                .attr("fill", "#4269d0")
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
                .style("cursor", "ns-resize");

            svg.append("text")
                .attr("x", width - margin.right).attr("y", 14)
                .attr("text-anchor", "end").attr("fill", "#4269d0")
                .style("cursor", "pointer").style("text-decoration", "underline")
                .text("reset")
                .on("click", () => { probs = original.slice(); redraw(); commit(); });

            function redraw() {
                bars.attr("y", (i) => y(probs[i])).attr("height", (i) => ih - y(probs[i]));
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
                const vars = {...model.get("variables")};
                vars.data = d3.range(n).map((i) => ({i: data[i].i, p: probs[i]}));
                vars.weights = probs.slice();
                vars.faces = data.map((d) => d.i);
                model.set("variables", vars);
                model.save_changes();
            }

            const drag = d3.drag()
                .container(() => g.node())
                .on("start", (event) => { if (event.sourceEvent) event.sourceEvent.preventDefault(); })
                .on("drag", (event, i) => setBar(i, y.invert(event.y)))
                .on("end", commit);

            hits.call(drag);

            return svg.node();
            """,
            variables={
                "data": data,
                "editMode": mode,
                "faces": [row["i"] for row in data],
                "weights": [row["p"] for row in data],
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
        # its parents. A root's distribution is simply itself.
        if self._fn is None:
            return self
        values = [assignment[parent] for parent in self._parents]
        return _coerce_dice(self._fn(*values))

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


class World:
    """A shared exact joint over the variables created within it.

    Yielded by ``with model() as m:``. Tracks the model's source variables
    (roots + dependents, in creation order so a child's parents are bound
    before it) and any named derived expressions, and answers marginal /
    conditional queries by enumerating the joint.
    """

    def __init__(self) -> None:
        self.sources: "list[Dice[Any]]" = []
        self.derived: "list[Dice[Any]]" = []

    def _register(self, variable: "Dice[Any]") -> None:
        self.sources.append(variable)

    def _register_derived(self, variable: "Dice[Any]") -> None:
        self.derived.append(variable)

    def joint(self) -> "list[tuple[dict[Dice[Any], Any], Weight]]":
        rows: "list[tuple[dict[Dice[Any], Any], Weight]]" = [({}, 1)]
        for source in self.sources:
            expanded: "list[tuple[dict[Dice[Any], Any], Weight]]" = []
            for assignment, weight in rows:
                for value, probability in source._dist_for(assignment).probs.items():
                    expanded.append(({**assignment, source: value}, weight * probability))
            rows = expanded
        return rows

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
        evidence is the point-mass special case. With several soft-evidence
        variables the updates are applied in turn (sequential Jeffrey
        conditioning).
        """
        # Enumerate with the model deactivated, so dice built by conditional
        # functions (and the result itself) are not registered as variables.
        token = _active_world.set(None)
        try:
            rows = self.joint()
            if given is not None:
                rows = [(a, w) for a, w in rows if given._eval(a)]
            for node, dist in (evidence or {}).items():
                marginal: "defaultdict[Any, Weight]" = defaultdict(int)
                for assignment, weight in rows:
                    marginal[node._eval(assignment)] += weight
                total = sum(marginal.values())
                if total == 0:
                    raise ValueError("evidence has zero probability")
                scale = sum(dist.values())
                target_prob = {k: v / scale for k, v in dist.items()}
                # reweight each atom so node's marginal matches target_prob
                rows = [
                    (assignment, weight * target_prob.get(value, 0) * total / marginal[value])
                    for assignment, weight in rows
                    for value in (node._eval(assignment),)
                    if marginal[value] > 0
                ]
            weights: "defaultdict[Any, Weight]" = defaultdict(int)
            for assignment, weight in rows:
                weights[target._eval(assignment)] += weight
            if sum(weights.values()) == 0:
                raise ValueError("evidence has zero probability")
            return Dice._raw(dict(weights))
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

    def show(
        self,
        given: "Dice[Any] | None" = None,
        inputs: "Mapping[Dice[Any], Any] | None" = None,
        editable: "Mapping[Dice[Any], Any] | None" = None,
        evidence: "Mapping[Dice[Any], Any] | None" = None,
    ) -> Any:
        """Render the model as a DAG of per-variable charts (marimo only).

        Each source and each named derived variable becomes a node,
        labelled with its Python name, with an arrow for every dependency,
        via wigglystuff's ``WidgetDAG``.

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

        ``evidence`` maps *any* variable to a :meth:`editor` widget and treats
        the dragged distribution as soft (Jeffrey) *evidence* on that node —
        not a change to the model, but an observation to condition on. Belief
        flows both ways, so dragging a child updates its parents' posteriors
        too. Under the hood every other node is queried with this soft
        evidence (see :meth:`query`).
        """
        import marimo as mo
        from wigglystuff import WidgetDAG

        graph_nodes = [*self.sources, *self.derived]
        tracked_ids = {id(node) for node in graph_nodes}
        names = {id(node): (node._name or f"var{index}") for index, node in enumerate(graph_nodes)}

        editable_by_id = {id(node): widget for node, widget in (editable or {}).items()}
        evidence_by_id = {id(node): widget for node, widget in (evidence or {}).items()}
        widget_by_id = {**editable_by_id, **evidence_by_id}
        # dragged evidence editors become soft evidence for querying the rest
        soft = {node: Dice.from_editor(widget).probs for node, widget in (evidence or {}).items()}

        nodes: dict[str, Any] = {
            names[id(node)]: (
                widget_by_id[id(node)]
                if id(node) in widget_by_id
                else mo.ui.anywidget(self.query(node, given, soft).prob_chart())
            )
            for node in graph_nodes
        }

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
