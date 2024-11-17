<img src="imgs/dice.png" alt="dicekit logo" width="125" align="right"/>

### dicekit

> Just some nice objects in Python

## Install

```bash
uv pip install dicekit
```

## Usage

There are many ways to construct a `Dice` object, but a simple one is to specify the number of sides.

```python
from dicekit import Dice

d6 = Dice.from_sides(6)
d8 = Dice.from_sides(8)
```

The nice thing about these objects is their representation in the notebook.

![Dice output](imgs/overview.png)

To learn more, we recommend checking out [this Marimo notebook](https://koaning.github.io/dicekit/). It shows the implementation, but also the flexibility of the objects. You can also copy the notebook and play around with it straight from the browser.
