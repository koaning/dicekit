 

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


def p(expression):
    return expression.probs[True]

def exp(dice):
    return sum(i * p for i, p in dice.probs.items())

def var(dice):
    return sum(p * (i - exp(dice))**2 for i, p in dice.probs.items())
