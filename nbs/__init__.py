import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## `dicekit`

    The goal of `dicekit` is two-fold. 

    - The first is to offer a simple library to interact with dice.
    - The second is to explore how Marimo may give us a domain specific environment to work/develop with dice. We could just work on a domain specific languge, but what if we can adapt the environment around the language a bit more so that it promotes interactivity and curiosity a bit more?

    ## `Dice` objects 

    The main object that you will interact with is the `Dice` object.

    ```python
    from dicekit import Dice
    ```

    These objects give you a flexible way to declare dice, and they also come with a convient visualisation of the probability distribution that they represent.
    """
    )
    return


@app.cell
def _(Dice):
    Dice.from_sides(6)
    return


@app.cell
def _(mo):
    mo.md("""These dice can be stored into variables, but they can also be added/subtracted as if they were normal Python numbers.""")
    return


@app.cell
def _(Dice):
    d6 = Dice.from_sides(6)
    d8 = Dice.from_sides(8)

    d6 + d8
    return (d6,)


@app.cell
def _(mo):
    mo.md("""These dice have a bunch of utilities attached that make it easy to get the distribution that you are interested in. For example, what if you are interested in the maximum of two dice rolls? You can use the `.out_of` method for that.""")
    return


@app.cell
def _(mo):
    sides_slider = mo.ui.slider(1, 100, 1, value=20, label="sides")
    sides_slider
    return (sides_slider,)


@app.cell
def _(Dice, mo, sides_slider):
    mo.hstack([
        Dice.from_sides(sides_slider.value), 
        Dice.from_sides(sides_slider.value).out_of(2, max),
        Dice.from_sides(sides_slider.value).out_of(3, max),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    When you have dice, you're typically also interested in their probabilities. You can use comparison operators for this, and we also have a convience function to give you the probability that you're interested in.

    ```python
    from dicekit import p, exp, var
    ```
    """
    )
    return


@app.cell
def _(Dice, p):
    d20 = Dice.from_sides(20)

    # DnD rules, how much more likely are you to win when you are at advantage?
    p(d20.out_of(2, max) > d20.out_of(2, min))
    return (d20,)


@app.cell
def _(mo):
    mo.md(r"""Under the hood, a comparison operator merely generates another `Dice`.""")
    return


@app.cell
def _(d20):
    d20.out_of(2, max) > d20.out_of(2, min)
    return


@app.cell
def _(mo):
    mo.md(r"""There are also some other convenience functions available such as `exp` and `var`.""")
    return


@app.cell
def _(d6, exp, var):
    exp(d6), var(d6)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Implementation 

    Fun fact: this library is completely defined and maintained from a Marimo notebook! The convenience here is that I can glance at the notebook to confirm if it is working and it also reinforces a good documentation practice. Feel free to explore [the repository](https://github.com/koaning/dicekit) to learn out it is set up. 

    You can also inspect the full implementation below.
    """
    )
    return


@app.cell
def _():
    ## Export 

    import altair as alt 
    import pandas as pd
    import marimo as mo
    import random 
    from collections import Counter

    class Dice:
        def __init__(self, probs):
            self.probs = probs

        @classmethod
        def from_sides(self, n=6):
            return Dice({i: 1/n for i in range(1, n+1)})

        @classmethod
        def from_numbers(self, *args):
            c = Counter(args)
            return Dice({k: v/len(args) for k, v in args})

        def roll(self, n=1):
            return random.choices(list(self.probs.keys()), weights=list(self.probs.values()), k=n)

        def operate(self, other, operator):
            if isinstance(other, (float, int)):
                other = Dice({other: 1})
            new_probs = {}
            for s1, p1 in self.probs.items():
                for s2, p2 in other.probs.items():
                    new_key = operator(s1, s2)
                    if new_key not in new_probs:
                        new_probs[new_key] = 0
                    new_probs[new_key] += p1 * p2
            return Dice(new_probs)

        def filter(self, func):
            new_probs = {k: v for k, v in self.probs.items() if func(k)}
            total_prob = sum(new_probs.values())
            return Dice({k: v/total_prob for k, v in new_probs.items()})

        def _repr_html_(self):
            return mo.as_html(self.prob_chart()).text

        def prob_chart(self):
            df = pd.DataFrame([{"i": k, "p": v} for k, v in self.probs.items()])
            return (
                alt.Chart(df)
                  .mark_bar()
                  .encode(x="i", y="p")
                  .properties(title="Dice with probabilities:", width=120, height=120)
            )

        def out_of(self, n=2, func=max):
            current = Dice(self.probs)
            for i in range(n - 1):
                current = current.operate(current, operator=lambda a, b: func(a, b))
            return current

        def __add__(self, other):
            return self.operate(other, lambda a,b: a + b)

        def __sub__(self, other):
            return self.operate(other, lambda a,b: a - b)

        def __mul__(self, other):
            return self.operate(other, lambda a,b: a * b)

        def __le__(self, other):
            return self.operate(other, lambda a,b: a <= b)

        def __lt__(self, other):
            return self.operate(other, lambda a,b: a < b)

        def __ge__(self, other):
            return self.operate(other, lambda a,b: a >= b)

        def __gt__(self, other):
            return self.operate(other, lambda a,b: a > b)

        def __len__(self):
            return len(self.probs)
    return Dice, mo


@app.cell
def _():
    ## Export

    def p(expression):
        return expression.probs[True]

    def exp(dice):
        return sum(i * p for i, p in dice.probs.items())

    def var(dice):
        return sum(p * (i - exp(dice))**2 for i, p in dice.probs.items())
    return exp, p, var


if __name__ == "__main__":
    app.run()
