__all__ = ['Event', 'Evidence', 'Node', 'Factor', 'DAG']


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
