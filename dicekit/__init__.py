 

import altair as alt 
import pandas as pd
import marimo as mo
import random 
from collections import Counter

class Dice:
    """
    A class representing dice with arbitrary probability distributions.

    This class allows creating, manipulating, and visualizing dice with
    custom probability distributions for each face value.
    """

    def __init__(self, probs):
        """
        Initialize a Dice object with given probabilities.

        Parameters:
            probs (dict): A dictionary mapping face values to probabilities
        """
        self.probs = probs

    @classmethod
    def from_sides(cls, n=6):
        """
        Create a fair dice with n sides.

        Parameters:
            n (int): Number of sides, default is 6

        Returns:
            Dice: A fair dice with n sides
        """
        return cls({i: 1/n for i in range(1, n+1)})

    @classmethod
    def from_numbers(cls, *args):
        """
        Create a dice from a sequence of numbers with frequencies.

        Parameters:
            *args: Variable length list of numbers

        Returns:
            Dice: A dice with probabilities based on the frequency of each number
        """
        c = Counter(args)
        return cls({k: v/len(args) for k, v in args.items()})

    def roll(self, n=1):
        """
        Simulate rolling the dice n times.

        Parameters:
            n (int): Number of rolls, default is 1

        Returns:
            list: Results of the dice rolls
        """
        return random.choices(list(self.probs.keys()), weights=list(self.probs.values()), k=n)

    def operate(self, other, operator):
        """
        Apply an operation between this dice and another dice or number.

        Parameters:
            other: Another Dice object or a number
            operator: A function that takes two arguments and returns a result

        Returns:
            Dice: A new dice representing the distribution of the operation's results
        """
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
        """
        Create a new dice by filtering face values based on a function.

        Parameters:
            func: A function that returns True for values to keep

        Returns:
            Dice: A new dice with only the filtered values, renormalized
        """
        new_probs = {k: v for k, v in self.probs.items() if func(k)}
        total_prob = sum(new_probs.values())
        return Dice({k: v/total_prob for k, v in new_probs.items()})

    def _repr_html_(self):
        """
        Return HTML representation of the dice for display in notebooks.

        Returns:
            str: HTML string for rendering
        """
        return mo.as_html(self.prob_chart()).text

    def prob_chart(self):
        """
        Create a visualization of the dice probability distribution.

        Returns:
            alt.Chart: An Altair chart showing the dice distribution
        """
        df = pd.DataFrame([{"i": k, "p": v} for k, v in self.probs.items()])
        return (
            alt.Chart(df)
              .mark_bar()
              .encode(
                x="i", 
                y="p",
                tooltip=[
                    alt.Tooltip("i", title="Value"),
                    alt.Tooltip("p", title="Probability", format=".3f")
                ]
              )
              .properties(title="Dice with probabilities:", width=120, height=120)
              .interactive()
        )

    def out_of(self, n=2, func=max):
        """
        Create a dice representing the result of applying a function to n rolls.

        Parameters:
            n (int): Number of dice to roll, default is 2
            func: Function to apply to the rolls (e.g., max, min), default is max

        Returns:
            Dice: A new dice representing the distribution of the function's results
        """
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
    """
    Returns the probability of a True outcome from a dice expression.

    Parameters:
        expression: A Dice object representing a boolean comparison

    Returns:
        float: The probability of the True outcome
    """
    return expression.probs[True]

def exp(dice):
    """
    Calculates the expected value (mean) of a dice.

    Parameters:
        dice: A Dice object

    Returns:
        float: The expected value of the dice
    """
    return sum(i * p for i, p in dice.probs.items())

def var(dice):
    """
    Calculates the variance of a dice.

    Parameters:
        dice: A Dice object

    Returns:
        float: The variance of the dice
    """
    return sum(p * (i - exp(dice))**2 for i, p in dice.probs.items())

def ordered(*dice_in):
    """
    Return dice that represent order statistics. Highest first.

    Takes multiple dice objects and returns new dice objects that represent
    the ordered outcomes (e.g., highest roll, second highest roll, etc.).

    Parameters:
        *dice_in: Variable number of Dice objects

    Returns:
        list: A list of Dice objects representing order statistics
    """
    result = {}
    for _i in product(*[d.probs.items() for d in dice_in]):
        eyes = tuple(sorted([_[0] for _ in _i], reverse=True))
        prob = reduce(lambda a, b: a * b, [_[1] for _ in _i])
        if eyes not in result:
            result[eyes] = 0
        result[eyes] += prob

    dice_out = []
    for _j in range(len(dice_in)):
        new_dice = {}
        for keys, pval in result.items():
            if keys[_j] not in new_dice:
                new_dice[keys[_j]] = 0
            new_dice[keys[_j]] += pval
        dice_out.append(Dice(new_dice))
    return dice_out
