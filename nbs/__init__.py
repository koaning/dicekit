# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hastyplot==0.4.1",
#     "marimo>=0.23.9",
#     "pandas==3.0.3",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md("""
    # `dicekit`

    The goal of `dicekit` is two-fold.

    - The first is to offer a simple library to interact with dice.
    - The second is to explore how Marimo may give us a domain specific environment to work/develop with dice. We could just work on a domain specific languge, but what if we can adapt the environment around the language a bit more so that it promotes interactivity and curiosity a bit more?

    ## `Dice` objects

    The main object that you will interact with is the `Dice` object.

    ```python
    from dicekit import Dice
    ```

    These objects give you a flexible way to declare dice, and they also come with a convient visualisation of the probability distribution that they represent.
    """)
    return


@app.cell
def _(Dice):
    Dice.from_sides(6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    These dice can be stored into variables, but they can also be added/subtracted as if they were normal Python numbers.
    """)
    return


@app.cell
def _(Dice):
    d6 = Dice.from_sides(6)
    d8 = Dice.from_sides(8)

    d6 + d8
    return (d6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    By default a `Dice` renders the probability of each outcome via `.prob_chart()`. When you'd rather see the probability of rolling *at most* a given value, you can reach for the cumulative distribution function with `.cdf_chart()`.
    """)
    return


@app.cell
def _(d6):
    (d6 + d6).cdf_chart()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    These dice have a bunch of utilities attached that make it easy to get the distribution that you are interested in. For example, what if you are interested in the maximum of two dice rolls? You can use the `.out_of` method for that.
    """)
    return


@app.cell
def _(mo):
    sides_slider = mo.ui.slider(1, 100, 1, value=20, label="sides")
    sides_slider
    return (sides_slider,)


@app.cell
def _(Dice, mo, sides_slider):
    mo.hstack(
        [
            Dice.from_sides(sides_slider.value),
            Dice.from_sides(sides_slider.value).out_of(2, max),
            Dice.from_sides(sides_slider.value).out_of(3, max),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    You can also approximate a continuous distribution with a dice via `Dice.from_dist`. It slices equiprobable quantiles out of any object that exposes an inverse-CDF, such as a `statistics.NormalDist` (`.inv_cdf`) or a frozen `scipy.stats` distribution (`.ppf`).

    Two keyword arguments steer the approximation:

    - `n` (default `6`): the number of equiprobable faces, placed at the midpoint of each quantile bin. The more faces you ask for, the closer the approximation.
    - `quantiles`: an explicit sequence of probabilities in `(0, 1)` to slice at, instead of `n` evenly spaced midpoints. Every resulting face stays equiprobable.
    """)
    return


@app.cell
def _(Dice):
    from statistics import NormalDist

    Dice.from_dist(NormalDist(0, 1), n=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Because `from_dist` only needs an object with a `.ppf` (or `.inv_cdf`) method, you can also reach for `numpy`. The exponential distribution has a closed-form inverse-CDF, `ppf(q) = -scale * log(1 - q)`, so a tiny wrapper is all it takes.
    """)
    return


@app.cell
def _(Dice):
    import numpy as np


    class Exponential:
        def __init__(self, scale=1.0):
            self.scale = scale

        def ppf(self, q):
            return float(-self.scale * np.log(1 - q))


    Dice.from_dist(Exponential(scale=2.0), n=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    When you have dice, you're typically also interested in their probabilities. You can use comparison operators for this, and we also have a convience function to give you the probability that you're interested in.

    ```python
    from dicekit import p, exp, var
    ```
    """)
    return


@app.cell
def _(Dice, p):
    d20 = Dice.from_sides(20)

    # DnD rules, how much more likely are you to win when you are at advantage?
    p(d20.out_of(2, max) > d20)
    return (d20,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Under the hood, a comparison operator merely generates another `Dice`.
    """)
    return


@app.cell
def _(d20):
    d20.out_of(2, max) > d20
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are also some other convenience functions available such as `exp` and `var`.
    """)
    return


@app.cell
def _(d6, exp, var):
    exp(d6), var(d6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact probabilities with `Fraction`

    By default the probabilities are floating point numbers, but if you'd rather work with *exact* values you can build a `Dice` from `fractions.Fraction` weights. The probabilities stay exact through arithmetic and helpers like `exp`, while the charts still render just fine.
    """)
    return


@app.cell
def _(Dice, Fraction, exp):
    fair = Dice({i: Fraction(1, 6) for i in range(1, 7)})

    # exact arithmetic: exp(2d6) is Fraction(7, 1), not 6.999999...
    exact = fair + fair
    exp(exact)
    return (exact,)


@app.cell
def _(exact):
    exact
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Order Statistics

    The library also comes with an `ordered` function that lets you calculate order statistics from a set of dice.

    ```python
    from dicekit import ordered
    ```

    In the example below we explore a situation from the Risk boardgame. There are three attacking armies and two defending ones.
    """)
    return


@app.cell
def _(d6, mo, ordered):
    a1, a2, a3 = ordered(d6, d6, d6)
    d1, d2 = ordered(d6, d6)

    mo.hstack([a1 > d1, a2 > d2])
    return a1, a2, d1, d2


@app.cell(hide_code=True)
def _(a1, a2, d1, d2, mo, p):
    mo.md(f"""
    The probability that the best attacking army defeats the defending army is {p(a1 > d1):.3f}. But the second best attacking army has a bigger chance: {p(a2 > d2):.3f}. These numbers are confirmed with the simulation below.
    """)
    return


@app.cell
def _(a1, d6):
    a1, d6.out_of(3, max)
    return


@app.cell
def _(random):
    # Simulation, just in case to check. This is not part of the library.
    from statistics import mean


    def simulate(n=10_000):
        results = []
        for _ in range(n):
            a1, a2, a3 = sorted([random.randint(1, 6) for _ in range(3)], reverse=True)
            d1, d2 = sorted([random.randint(1, 6) for _ in range(2)], reverse=True)
            results.append((a1 > d1, a2 > d2))
        return mean([_[0] for _ in results]), mean([_[1] for _ in results])


    simulate()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `Vase` to `Dice`

    ```python
    from dicekit import Vase
    ```

    We also offer a utility class called a `Vase` to mimic situations when you're trying to grab items from a bag and you'd like to know things related to how likely it is to grab combinations of items.
    """)
    return


@app.cell
def _(Vase):
    Vase.from_counts(a=3, b=3, c=1).take(2, replace=True, ordered=True)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    Fun fact: this library is completely defined and maintained from a Marimo notebook! The convenience here is that I can glance at the notebook to confirm if it is working and it also reinforces a good documentation practice. Feel free to explore [the repository](https://github.com/koaning/dicekit) to learn out it is set up.

    You can also inspect the full implementation below.
    """)
    return


@app.cell
def _():
    ## Export

    import pandas as pd
    import marimo as mo
    from hastyplot import qplot
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

        def _repr_html_(self) -> str:
            """
            Return HTML representation of the dice for display in notebooks.

            Returns:
                str: HTML string for rendering
            """
            return mo.as_html(self.prob_chart()).text

        def prob_chart(self) -> Any:
            """
            Create a visualization of the dice probability distribution.

            Returns:
                alt.Chart: An Altair chart showing the probability mass function
            """
            df = pd.DataFrame(
                [{"i": _plot_value(k), "p": _plot_value(v)} for k, v in self.probs.items()]
            )
            return qplot(
                df,
                "i",
                "p",
                mark="bar",
                tooltip=["i", "p"],
                title="Dice with probabilities:",
                width=120,
                height=120,
                theme="default",
            )

        def cdf_chart(self) -> Any:
            """
            Create a visualization of the cumulative distribution function.

            Returns:
                alt.Chart: An Altair step chart showing the cumulative probability
            """
            cumulative = 0
            rows = []
            for outcome, probability in sorted(self.probs.items()):
                cumulative += probability
                rows.append({"i": _plot_value(outcome), "p": _plot_value(cumulative)})
            df = pd.DataFrame(rows)
            return qplot(
                df,
                "i",
                "p",
                mark="step",
                tooltip=["i", "p"],
                title="Cumulative distribution:",
                width=120,
                height=120,
                theme="default",
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
        internal = {"from_numbers", "from_sides", "from_dist", "from_function", "take", "_raw", "_make", "_derive", "_coerce_dice"}
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

        def query(self, target: "Dice[Any]", given: "Dice[Any] | None" = None) -> "Dice[Any]":
            # Enumerate with the model deactivated, so dice built by conditional
            # functions (and the result itself) are not registered as variables.
            token = _active_world.set(None)
            try:
                weights: "defaultdict[Any, Weight]" = defaultdict(int)
                for assignment, weight in self.joint():
                    if given is not None and not given._eval(assignment):
                        continue
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

        def show(self, given: "Dice[Any] | None" = None, inputs: "Mapping[Dice[Any], Any] | None" = None) -> Any:
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
            """
            import marimo as mo
            from wigglystuff import WidgetDAG

            graph_nodes = [*self.sources, *self.derived]
            tracked_ids = {id(node) for node in graph_nodes}
            names = {id(node): (node._name or f"var{index}") for index, node in enumerate(graph_nodes)}

            nodes: dict[str, Any] = {
                names[id(node)]: mo.ui.altair_chart(
                    self.query(node, given).prob_chart(),
                    chart_selection=False,
                    legend_selection=False,
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

    return Any, Callable, Counter, Dice, Fraction, Hashable, Mapping, Sequence, Weight, World, cast, dataclass, defaultdict, mo, model, mul, permutations, product, random, reduce


@app.cell
def _(Counter, Dice, Sequence, permutations, product):
    ## Export


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

    return (Vase,)


@app.cell
def _(Any, Dice, Hashable, Sequence, Weight, cast):
    ## Export

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

    return exp, ordered, p, var


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Puzzle models

    A `Dice` created inside `with model():` *is* a random variable: it plots on
    its own, and expressions across variables resolve against one shared exact
    joint. Roots are bare dice; `Dice.from_function(fn, parents=[...])` adds a
    dependent one. Below is Monty Hall — switching wins two out of three times.
    """)
    return


@app.cell
def _(Dice, model, p):
    with model() as monty:
        car = Dice.from_numbers(1, 2, 3)                     # where the car is
        pick = Dice.from_numbers(1, 2, 3)                    # your first choice
        # the host opens a door that is neither the car nor your pick
        opened = Dice.from_function(
            lambda c, g: Dice.from_numbers(*[d for d in (1, 2, 3) if d != c and d != g]),
            parents=[car, pick],
        )
        switch = Dice.from_numbers(True, False)              # your strategy
        # switching wins iff your first pick was wrong
        win = (switch & (car != pick)) | (~switch & (car == pick))

    # switching wins 2/3, staying wins 1/3
    p(win, given=switch == True), p(win, given=switch == False)
    return (monty,)


@app.cell
def _(monty):
    monty.show()  # the whole model as a DAG of per-variable marginal charts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evidence and controls

    `show(given=...)` charts every node's *posterior* under some evidence, so
    you can watch it propagate. A slider lives in its own cell and is read via
    `.value` inside the variable it feeds; `show(inputs=...)` draws it as an
    upstream node. Drag it and marimo recomputes the whole model.
    """)
    return


@app.cell
def _(mo):
    prevalence = mo.ui.slider(0.001, 0.3, value=0.01, step=0.001, label="prevalence")
    prevalence
    return (prevalence,)


@app.cell
def _(Dice, model, prevalence):
    with model() as screen:
        sick = Dice({"present": prevalence.value, "absent": 1 - prevalence.value})
        swab = Dice.from_function(
            lambda s: {"pos": 0.9, "neg": 0.1} if s == "present" else {"pos": 0.05, "neg": 0.95},
            parents=[sick],
        )
    return screen, sick, swab


@app.cell
def _(prevalence, screen, sick, swab):
    # posterior charts given a positive swab, with the slider wired in upstream
    screen.show(given=swab == "pos", inputs={sick: prevalence})
    return


if __name__ == "__main__":
    app.run()
