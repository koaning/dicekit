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
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from collections import Counter
    from itertools import product, permutations
    from functools import reduce
    from typing import Any, Self, TypeAlias, cast

    Weight: TypeAlias = int | float

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
            """
            self.probs = {k: v / sum(probs.values()) for k, v in probs.items()}

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

        def roll(self, n: int = 1) -> list[T]:
            """
            Simulate rolling the dice n times.

            Parameters:
                n (int): Number of rolls, default is 1

            Returns:
                list: Results of the dice rolls
            """
            return random.choices(list(self.probs.keys()), weights=list(self.probs.values()), k=n)

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
            return Dice(new_probs)

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
                    other = Dice({other: 1})
                except TypeError as exc:
                    raise TypeError("operations require a Dice or hashable value") from exc
            other = cast(Dice[U], other)
            new_probs: dict[V, float] = {}
            for s1, p1 in self.probs.items():
                for s2, p2 in other.probs.items():
                    new_key = operator(s1, s2)
                    if new_key not in new_probs:
                        new_probs[new_key] = 0
                    new_probs[new_key] += p1 * p2
            return Dice(new_probs)

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
            return Dice({k: v / total_prob for k, v in new_probs.items()})

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
            df = pd.DataFrame([{"i": k, "p": v} for k, v in self.probs.items()])
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
                rows.append({"i": outcome, "p": cumulative})
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
            result: dict[U, float] = {}
            dice_in = [self] * n
            for _i in product(*[d.probs.items() for d in dice_in]):
                values = [_[0] for _ in _i]
                prob = reduce(lambda a, b: a * b, [_[1] for _ in _i])
                outcome = func(values)
                if outcome not in result:
                    result[outcome] = 0
                result[outcome] += prob
            return Dice(result)

        def __add__(self, other: Any) -> "Dice[Any] | Self":
            if not isinstance(other, Dice) and other == 0:
                return self
            return self.operate(other, lambda a, b: a + b)

        def __radd__(self, other: Any) -> "Dice[Any] | Self":
            if not isinstance(other, Dice) and other == 0:
                return self
            return self.operate(other, lambda a, b: a + b)

        def __sub__(self, other: Any) -> "Dice[Any]":
            return self.operate(other, lambda a, b: a - b)

        def __rsub__(self, other: Any) -> "Dice[Any]":
            return self.operate(other, lambda a, b: b - a)

        def __mul__(self, other: Any) -> "Dice[Any]":
            return self.operate(other, lambda a, b: a * b)

        def __rmul__(self, other: Any) -> "Dice[Any]":
            return self.operate(other, lambda a, b: a * b)

        def __le__(self, other: Any) -> "Dice[bool]":
            return self.operate(other, lambda a, b: a <= b)

        def __lt__(self, other: Any) -> "Dice[bool]":
            return self.operate(other, lambda a, b: a < b)

        def __ge__(self, other: Any) -> "Dice[bool]":
            return self.operate(other, lambda a, b: a >= b)

        def __gt__(self, other: Any) -> "Dice[bool]":
            return self.operate(other, lambda a, b: a > b)

        def __eq__(self, other: Any) -> "Dice[bool]":  # type: ignore[override]
            return self.operate(other, lambda a, b: a == b)

        def __ne__(self, other: Any) -> "Dice[bool]":  # type: ignore[override]
            return self.operate(other, lambda a, b: a != b)

        def __len__(self) -> int:
            return len(self.probs)

    return Any, Callable, Counter, Dice, Hashable, Mapping, Sequence, Weight, cast, mo, permutations, product, random, reduce


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
def _(Any, Callable, Dice, Hashable, Sequence, Weight, cast, product, reduce):
    ## Export


    def p(expression: "Dice[bool]") -> float:
        """
        Returns the probability of a True outcome from a dice expression.

        Parameters:
            expression: A Dice object representing a boolean comparison

        Returns:
            float: The probability of the True outcome
        """
        return expression.probs.get(True, 0)


    def exp(dice: "Dice[int | float]") -> float:
        """
        Calculates the expected value (mean) of a dice.

        Parameters:
            dice: A Dice object

        Returns:
            float: The expected value of the dice
        """
        return sum(i * p for i, p in dice.probs.items())


    def var(dice: "Dice[int | float]") -> float:
        """
        Calculates the variance of a dice.

        Parameters:
            dice: A Dice object

        Returns:
            float: The variance of the dice
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

        new_probs: dict[T, float] = {}
        for die, weight in zip(dice, weights):
            for outcome, probability in die.probs.items():
                contribution = probability * weight / total_weight
                new_probs[outcome] = new_probs.get(outcome, 0) + contribution
        return Dice(new_probs)


    def _slice_limit(length: int, stop: int) -> int:
        return len(range(length)[:stop])


    def _threshold_probabilities[T: Hashable](die: "Dice[T]", thresholds: Sequence[T]) -> dict[T, float]:
        """
        Return P(die >= threshold) for each threshold.
        """
        probabilities: dict[T, float] = {}
        cumulative = 0
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

        dice_out: list[dict[T, float]] = [{} for _ in range(n_outputs)]
        higher_threshold_survival: list[float] = [0.0] * (n_outputs + 1)

        for threshold in thresholds:
            pass_count_probs: list[float] = [1.0] + [0.0] * n_outputs

            for probability in probabilities:
                p_ge = probability[threshold]
                p_lt = 1 - p_ge
                new_pass_count_probs: list[float] = [0.0] * (n_outputs + 1)

                for pass_count, pass_count_probability in enumerate(pass_count_probs):
                    if pass_count_probability == 0:
                        continue
                    new_pass_count_probs[pass_count] += pass_count_probability * p_lt
                    capped_pass_count = min(pass_count + 1, n_outputs)
                    new_pass_count_probs[capped_pass_count] += (
                        pass_count_probability * p_ge
                    )

                pass_count_probs = new_pass_count_probs

            survival: list[float] = [0.0] * (n_outputs + 1)
            running = 0.0
            for pass_count in range(n_outputs, -1, -1):
                running += pass_count_probs[pass_count]
                survival[pass_count] = running

            for order_index in range(n_outputs):
                order = order_index + 1
                probability = survival[order] - higher_threshold_survival[order]
                if probability < 0 and abs(probability) < 1e-15:
                    probability = 0.0
                if probability != 0:
                    dice_out[order_index][threshold] = probability

            higher_threshold_survival = survival

        return [Dice(probs) for probs in dice_out]

    return exp, ordered, p, var


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
