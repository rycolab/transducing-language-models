"""Finite-state transducer (FST) data structure with composition and label pushing.

Provides the :class:`FST` class for building, inspecting, and composing
finite-state transducers.  An FST maps source strings to target strings via
labeled arcs.  Supports epsilon-free and epsilon-rich machines, composition
(``@``), projection, label pushing, reachability analysis, and Graphviz
visualization.

See the FSA acceptor for the acceptor (single-tape) counterpart.
"""

from __future__ import annotations

from collections import defaultdict, deque
from functools import cached_property
from itertools import zip_longest
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar, overload

from .fsa import FSA, EPSILON

from .util import Integerizer, State, Str

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class _EpsilonVariant:
    """Unique sentinel for composition epsilon filters (cannot collide with user labels)."""
    def __init__(self, name: str) -> None:
        self._name = name
    def __repr__(self) -> str:
        return self._name

ε_1 = _EpsilonVariant('ε₁')
ε_2 = _EpsilonVariant('ε₂')


eps = EPSILON


class FST(Generic[A, B]):
    """Finite-state transducer mapping source strings to target strings.

    An FST is a directed graph whose arcs carry input:output label pairs.
    A path from a start state to a stop state spells out a source string
    (concatenation of input labels) and a target string (concatenation of
    output labels).  The relation of the FST is the set of all such
    (source, target) pairs.

    Attributes:
        A: Set of input (source) symbols observed on arcs.
        B: Set of output (target) symbols observed on arcs.
        states: Set of all states.
        start: Set of initial states.
        stop: Set of final (accepting) states.
    """

    def __init__(self, start: Iterable[State] = (), arcs: Iterable[tuple[State, A, B, State]] = (), stop: Iterable[State] = ()) -> None:
        """Create an FST, optionally populating it from iterables.

        Args:
            start: Iterable of initial states.
            arcs: Iterable of ``(src, input_label, output_label, dst)`` tuples.
            stop: Iterable of final (accepting) states.
        """
        self.A: set[A] = set()
        self.B: set[B] = set()
        self.states: set[State] = set()
        self.delta: defaultdict[State, defaultdict[Any, defaultdict[Any, set[State]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.start: set[State] = set()
        self.stop: set[State] = set()
        self._arcs_i: dict[State, tuple[tuple[A, B, State], ...]] | None = None
        self._arcs_ix: dict[tuple[State, A], tuple[tuple[B, State], ...]] | None = None
        for i in start: self.add_start(i)
        for i in stop: self.add_stop(i)
        for i,a,b,j in arcs: self.add_arc(i,a,b,j)

    def __repr__(self) -> str:
        return f'{__class__.__name__}({len(self.states)} states)'

    def __str__(self) -> str:
        output: list[str] = []
        output.append('{')
        for p in self.states:
            output.append(f'  {p} \t\t({p in self.start}, {p in self.stop})')
            for a, b, q in self.arcs(p):
                output.append(f'    {a}:{b} -> {q}')
        output.append('}')
        return '\n'.join(output)

    def is_final(self, i: State) -> bool:
        """Return True if state ``i`` is a final (accepting) state."""
        return i in self.stop

    def add_arc(self, i: State, a: A, b: B, j: State) -> None:  # pylint: disable=arguments-renamed
        """Add arc from state ``i`` to state ``j`` with input label ``a`` and output label ``b``.

        Use ``EPSILON`` (the empty string ``''``) for either label to create
        epsilon transitions that consume no input or produce no output.
        States are created implicitly if they don't already exist.
        """
        self.states.add(i)
        self.states.add(j)
        self.delta[i][a][b].add(j)
        self.A.add(a)
        self.B.add(b)
        self._arcs_i = None   # invalidate arc indexes

    def add_start(self, q: State) -> None:
        """Mark state ``q`` as an initial state (creates it if needed)."""
        self.states.add(q)
        self.start.add(q)

    def add_stop(self, q: State) -> None:
        """Mark state ``q`` as a final (accepting) state (creates it if needed)."""
        self.states.add(q)
        self.stop.add(q)

    def ensure_trie_index(self) -> None:
        """Build secondary arc indexes for O(1) lookup by output or input.

        Builds three dicts (down from the previous four hash-map indexes):

        - ``_arcs_by_output[(i, y)]`` → ``tuple[(x, j), ...]``
        - ``_arcs_all[i]`` → ``tuple[(x, j), ...]``
        - ``_arcs_by_input[(i, x)]`` → ``tuple[(y, j), ...]``
        """
        if hasattr(self, '_arcs_by_output'):
            return
        arcs_by_output: dict[tuple[State, B], list[tuple[A, State]]] = {}
        arcs_all: dict[State, list[tuple[A, State]]] = {}
        arcs_by_input: dict[tuple[State, A], list[tuple[B, State]]] = {}
        for i in self.states:
            for x, y, j in self.arcs(i):
                arcs_by_output.setdefault((i, y), []).append((x, j))
                arcs_all.setdefault(i, []).append((x, j))
                arcs_by_input.setdefault((i, x), []).append((y, j))
        self._arcs_by_output = {k: tuple(v) for k, v in arcs_by_output.items()}
        self._arcs_all = {k: tuple(v) for k, v in arcs_all.items()}
        self._arcs_by_input = {k: tuple(v) for k, v in arcs_by_input.items()}


    def _build_arc_index(self) -> None:
        """Build flat arc indexes for O(1) lookup, called lazily on first arcs() call."""
        # state → tuple[(input, output, dest), ...]
        arcs_i: dict[State, tuple[tuple[A, B, State], ...]] = {}
        # (state, input) → tuple[(output, dest), ...]
        arcs_ix: dict[tuple[State, A], tuple[tuple[B, State], ...]] = {}
        for i, d in self.delta.items():
            all_arcs: list[tuple[A, B, State]] = []
            for a, A in d.items():
                by_input: list[tuple[B, State]] = []
                for b, B in A.items():
                    for j in B:
                        all_arcs.append((a, b, j))
                        by_input.append((b, j))
                arcs_ix[(i, a)] = tuple(by_input)
            arcs_i[i] = tuple(all_arcs)
        self._arcs_i = arcs_i
        self._arcs_ix = arcs_ix

    @overload
    def arcs(self, i: State) -> tuple[tuple[A, B, State], ...]: ...
    @overload
    def arcs(self, i: State, x: A) -> tuple[tuple[B, State], ...]: ...

    def arcs(self, i: State, x: A | None = None) -> tuple[tuple[A, B, State], ...] | tuple[tuple[B, State], ...]:
        """Iterate over arcs from state ``i``.

        With ``x=None``, yields ``(a, b, j)`` triples for all arcs from ``i``.
        With ``x`` provided, yields ``(b, j)`` pairs for arcs with input label ``x``.
        """
        if self._arcs_i is None:
            self._build_arc_index()
        assert self._arcs_i is not None and self._arcs_ix is not None
        if x is None:
            return self._arcs_i.get(i, ())
        else:
            return self._arcs_ix.get((i, x), ())

    def rename(self, f: Callable[[State], State]) -> FST[A, B]:
        """Return a new FST with states relabeled by ``f(state)``.

        If ``f`` is not injective, distinct states may merge.
        """
        m = self.spawn()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i in self.states:
            for a, b, j in self.arcs(i):
                m.add_arc(f(i), a, b, f(j))
        return m

    def map_labels(self, f: Callable[[A, B], tuple[Any, Any]]) -> FST[Any, Any]:
        "Transform arc labels by applying f(a, b) -> (a', b') to each arc."
        m = self.spawn(keep_init=True, keep_stop=True)
        for i in self.states:
            for a, b, j in self.arcs(i):
                a2, b2 = f(a, b)
                m.add_arc(i, a2, b2, j)
        return m

    def renumber(self) -> FST[A, B]:
        """Return a copy with states relabeled as consecutive integers."""
        return self.rename(Integerizer())  # pyright: ignore[reportUnknownArgumentType]

    def spawn(self, *, keep_init: bool = False, keep_arcs: bool = False, keep_stop: bool = False) -> FST[A, B]:
        """Create a new empty FST, optionally copying start states, arcs, and/or stop states."""
        m = self.__class__()
        if keep_init:
            for q in self.start:
                m.add_start(q)
        if keep_arcs:
            for i in self.states:
                for a, b, j in self.arcs(i):
                    m.add_arc(i, a, b, j)
        if keep_stop:
            for q in self.stop:
                m.add_stop(q)
        return m



    def __call__(self, x: Any, y: Any) -> Any:
        """
        Compute the total weight of x:y under the FST's weighted relation.  If one
        of x or y is None, we return the weighted language that is the cross
        section (we do so efficiently by representing it as a WFSA).
        """

        if x is not None and y is not None:
            if isinstance(x, str):
                x = FST.from_string(x)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if isinstance(y, str):
                y = FST.from_string(y)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return (x @ self @ y)

        elif x is not None and y is None:
            if isinstance(x, str):
                x = FST.from_string(x)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return (x @ self).project(1)

        elif x is None and y is not None:
            if isinstance(y, str):
                y = FST.from_string(y)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return (self @ y).project(0)

        else:
            return self

    @classmethod
    def from_string(cls, xs: Sequence[A]) -> FST[A, A]:
        """Build a linear identity FST that accepts exactly the string ``xs``.

        Each arc copies one symbol (input = output).  The resulting FST
        accepts ``xs`` as both its input and output language.
        """
        m = cls()
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add_arc(xs[:i], xs[i], xs[i], xs[:i+1])  # pyright: ignore[reportArgumentType]
        m.add_stop(xs)
        return m  # pyright: ignore[reportReturnType]

    @staticmethod
    def from_pairs(pairs: Sequence[tuple[Sequence[A], Sequence[B]]]) -> FST[A, B]:
        """Build an FST whose relation is the union of the given (input, output) pairs.

        Each ``(xs, ys)`` pair becomes a separate path.  Inputs and outputs
        of different lengths are padded with epsilon.
        """
        p: FST[A, B] = FST()
        p.add_start(0)
        p.add_stop(1)
        for i, (xs, ys) in enumerate(pairs):
            p.add_arc(0, EPSILON, EPSILON, (i, 0))  # pyright: ignore[reportArgumentType]
            for j, (x, y) in enumerate(zip_longest(xs, ys, fillvalue=EPSILON)):
                p.add_arc((i, j), x, y, (i, j + 1))  # pyright: ignore[reportArgumentType]
            p.add_arc((i, max(len(xs), len(ys))), EPSILON, EPSILON, 1)  # pyright: ignore[reportArgumentType]
        return p

    def project(self, axis: int) -> FSA[Any]:
        """
        Project the FST into a FSA when `component` is 0, we project onto the left,
        and with 1 we project onto the right.
        """
        assert axis in [0, 1]
        A: FSA[Any] = FSA()
        for i in self.states:
            for a, b, j in self.arcs(i):
                if axis == 0:
                    A.add_arc(i, a, j)
                else:
                    A.add_arc(i, b, j)
        for i in self.start:
            A.add_start(i)
        for i in self.stop:
            A.add_stop(i)
        return A


    @cached_property
    def T(self) -> FST[B, A]:
        "transpose swap left <-> right"
        return self.map_labels(lambda a, b: (b, a))

    def __matmul__(self, other: FST[B, C]) -> FST[A, C]:
        "Relation composition; may coerce `other` to an appropriate type if need be."

        if isinstance(other, FSA):
            other = FST.diag(other)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # minor efficiency trick: it's slightly more efficient to associate the composition as follows
        if len(self.states) < len(other.states):
            return (
                self._augment_epsilon_transitions(0)  # rename epsilons on the right
                ._compose(
                    epsilon_filter_fst(self.B),
                )  # this FST carefully combines the special epsilons
                ._compose(
                    other._augment_epsilon_transitions(1)
                )  # rename epsilons on the left
            )

        else:
            return (
                self._augment_epsilon_transitions(0)
                ._compose(  # rename epsilons on the right
                    epsilon_filter_fst(self.B)._compose(  # this FST carefully combines the special epsilons
                        other._augment_epsilon_transitions(1),
                    )
                )  # rename epsilons on the left
            )

    # TODO: use lazy machine pattern
    def _compose(self, other: FST[Any, Any]) -> FST[Any, Any]:
        """
        Implements the on-the-fly composition of the FST `self` with the FST `other`.

        Internal: both operands must already have epsilon-augmented labels
        (ε_1/ε_2) via ``_augment_epsilon_transitions()``.  Call ``__matmul__``
        for the public composition API.
        """

        C: FST[Any, Any] = FST()

        # index arcs in `other` to so that they are fast against later
        tmp: defaultdict[tuple[Any, Any], list[tuple[Any, Any]]] = defaultdict(list)
        for i in other.states:
            for a, b, j in other.arcs(i):
                tmp[i, a].append((b, j))

        visited: set[tuple[Any, Any]] = set()
        stack: list[tuple[Any, Any]] = []

        # add initial states
        for P in self.start:
            for Q in other.start:
                PQ = (P, Q)
                C.add_start(PQ)
                visited.add(PQ)
                stack.append(PQ)

        # traverse the machine using depth-first search
        while stack:
            P, Q = PQ = stack.pop()

            # (q,p) is simultaneously a final state in the respective machines
            if P in self.stop and Q in other.stop:
                C.add_stop(PQ)
                # Note: final states are not necessarily absorbing -> fall thru

            # Arcs of the composition machine are given by a cross-product-like
            # construction that matches an arc labeled `a:b` with an arc labeled
            # `b:c` in the left and right machines respectively.
            for a, b, Pʼ in self.arcs(P):
                for c, Qʼ in tmp[Q, b]:
                    assert b != EPSILON, (
                        "Matched on raw epsilon in _compose(); both operands must "
                        "have augmented epsilon labels (ε_1/ε_2) — use __matmul__ "
                        "instead of calling _compose() directly."
                    )

                    PʼQʼ = (Pʼ, Qʼ)

                    C.add_arc(PQ, a, c, PʼQʼ)

                    if PʼQʼ not in visited:
                        stack.append(PʼQʼ)
                        visited.add(PʼQʼ)

        return C

    # TODO: use lazy pattern here too.
    def _augment_epsilon_transitions(self, idx: int) -> FST[Any, Any]:
        """
        Augments the FST by changing the appropriate epsilon transitions to
        epsilon_1 or epsilon_2 transitions to be able to perform the composition
        correctly.  See Fig. 7 on p. 17 of Mohri, "Weighted Automata Algorithms".

        Args: `idx` (int): 0 if the FST is the first one in the composition, 1 otherwise.
        """
        assert idx in [0, 1]

        T: FST[Any, Any] = self.spawn(keep_init=True, keep_stop=True)

        for i in self.states:
            if idx == 0:
                T.add_arc(i, EPSILON, ε_1, i)  # pyright: ignore[reportArgumentType]
            else:
                T.add_arc(i, ε_2, EPSILON, i)  # pyright: ignore[reportArgumentType]
            for a, b, j in self.arcs(i):
                if idx == 0 and b == EPSILON:
                    b = ε_2
                elif idx == 1 and a == EPSILON:
                    a = ε_1
                T.add_arc(i, a, b, j)  # pyright: ignore[reportArgumentType]

        return T

    @classmethod
    def diag(cls, fsa: FSA[A]) -> FST[A, A]:
        """
        Convert a FSA A to diagonal relation T wich that T(x,x) = A(x) for all strings x.
        """
        assert isinstance(fsa, FSA), type(fsa)
        fst = cls()
        for i, a, j in fsa.arcs():
            fst.add_arc(i, a, a, j)  # pyright: ignore[reportArgumentType]
        for i in fsa.start:
            fst.add_start(i)
        for i in fsa.stop:
            fst.add_stop(i)
        return fst  # pyright: ignore[reportReturnType]

    def paths(self) -> Iterator[tuple[Any, ...]]:
        "Enumerate paths in the FST using breadth-first order."
        worklist: deque[tuple[Any, ...]] = deque()
        for i in self.start:
            worklist.append((i,))
        while worklist:
            path = worklist.popleft()
            i = path[-1]
            if self.is_final(i):
                yield path
            for a, b, j in self.arcs(i):
                worklist.append((*path, (a,b), j))

    def relation(self, max_length: int) -> Iterator[tuple[Str[A], Str[B]]]:
        "Enumerate string pairs in the relation of this FST up to length `max_length`."
        worklist: deque[tuple[State, Str[A], Str[B]]] = deque()
        worklist.extend([(i, (), ()) for i in self.start])
        while worklist:
            (i, xs, ys) = worklist.popleft()
            if self.is_final(i):
                yield xs, ys
            if len(xs) >= max_length or len(ys) >= max_length:
                continue
            for x, y, j in self.arcs(i):
                new_xs = (*xs, x) if x != EPSILON else xs
                new_ys = (*ys, y) if y != EPSILON else ys
                worklist.append((j, new_xs, new_ys))

    def transduce(self, input_seq: Sequence[A]) -> Iterator[Str[B]]:
        """Yield all distinct output strings for input_seq."""
        seen: set[Str[B]] = set()
        queue: deque[tuple[State, int, Str[B]]] = deque()
        visited: set[tuple[State, int, Str[B]]] = set()
        for s in self.start:
            entry = (s, 0, ())
            visited.add(entry)
            queue.append(entry)
        while queue:
            state, pos, output = queue.popleft()
            if pos == len(input_seq) and state in self.stop:
                if output not in seen:
                    seen.add(output)
                    yield output
            for a, b, j in self.arcs(state):
                if a == EPSILON:
                    new_pos = pos
                elif pos < len(input_seq) and a == input_seq[pos]:
                    new_pos = pos + 1
                else:
                    continue
                new_output = output if b == EPSILON else output + (b,)
                entry = (j, new_pos, new_output)
                if entry not in visited:
                    visited.add(entry)
                    queue.append(entry)

    def is_functional(self) -> tuple[bool, tuple[Str[A], Str[B], Str[B]] | None]:
        """Check whether this FST defines a (partial) function.

        An FST is functional if every input string maps to at most one
        output string.  Returns ``(True, None)`` if functional, or
        ``(False, (x, y1, y2))`` with a witness input ``x`` that produces
        two distinct outputs ``y1 != y2``.

        Note: a productive input-epsilon cycle (eps-input arcs that produce
        non-epsilon output) is a sufficient condition for non-functionality,
        since the cycle can be traversed any number of times yielding
        distinct outputs for the same input.

        Uses a product construction: two copies of the trimmed FST run
        on the same input while tracking the output-buffer difference.
        Terminates on all finite-state transducers (no length bound
        needed) by exploiting the twinning property: for a functional
        transducer the delay is bounded, and for a non-functional one
        we detect the pumping cycle.
        """
        t = self.trim()
        if not t.states:
            return (True, None)

        # --- Phase 1: Build product automaton (q1, q2) sharing input ---
        # Forward reachability from all start pairs.
        product_arcs: dict[tuple[State, State], list[tuple[Any, Any, Any, State, State]]] = {}
        product_final: set[tuple[State, State]] = set()

        fwd: set[tuple[State, State]] = set()
        queue: deque[tuple[State, State]] = deque()
        for s1 in t.start:
            for s2 in t.start:
                pair = (s1, s2)
                fwd.add(pair)
                queue.append(pair)

        while queue:
            q1, q2 = pair = queue.popleft()
            if q1 in t.stop and q2 in t.stop:
                product_final.add(pair)
            arcs: list[tuple[Any, Any, Any, State, State]] = []
            # Both copies take arcs with matching input (including eps-eps)
            for a, b1, j1 in t.arcs(q1):
                for a2, b2, j2 in t.arcs(q2):
                    if a != a2:
                        continue
                    arcs.append((a, b1, b2, j1, j2))
                    dest = (j1, j2)
                    if dest not in fwd:
                        fwd.add(dest)
                        queue.append(dest)
            # Copy 1 takes an epsilon-input arc, copy 2 stays
            for a, b1, j1 in t.arcs(q1):
                if a == EPSILON:
                    arcs.append((EPSILON, b1, EPSILON, j1, q2))
                    dest = (j1, q2)
                    if dest not in fwd:
                        fwd.add(dest)
                        queue.append(dest)
            # Copy 2 takes an epsilon-input arc, copy 1 stays
            for a2, b2, j2 in t.arcs(q2):
                if a2 == EPSILON:
                    arcs.append((EPSILON, EPSILON, b2, q1, j2))
                    dest = (q1, j2)
                    if dest not in fwd:
                        fwd.add(dest)
                        queue.append(dest)
            product_arcs[pair] = arcs

        # After trim(), every start state has a path to some stop state f.
        # Running both product copies from (s, s) on that path reaches (f, f),
        # so product_final is always non-empty.
        assert product_final, "product_final empty: (s,s) always reaches (f,f) after trim"

        # --- Phase 2: Backward reachability → co-accessible states ---
        rev: defaultdict[tuple[State, State], set[tuple[State, State]]] = defaultdict(set)
        for pair, arcs in product_arcs.items():
            for _, _, _, j1, j2 in arcs:
                rev[(j1, j2)].add(pair)

        co: set[tuple[State, State]] = set(product_final)
        queue: deque[tuple[State, State]] = deque(product_final)
        while queue:
            pair = queue.popleft()
            for prev in rev[pair]:
                if prev not in co:
                    co.add(prev)
                    queue.append(prev)

        # Only explore states on a start→final path in the product.
        trimmed_product = fwd & co

        # --- Phase 3: BFS with output-delay tracking ---
        # For a functional transducer the delay is bounded (twinning
        # property).  Bound B: if the delay exceeds |trimmed_product|*M,
        # some (q1,q2) pair was visited with strictly growing delay,
        # meaning a cycle is pumping.  Since every state in
        # trimmed_product is co-accessible, that cycle lies on a path
        # to acceptance → non-functional.
        delay_bound = max(len(trimmed_product), 1)

        visited: set[tuple[State, State, int, Str[Any]]] = set()
        worklist: deque[tuple[State, State, int, Str[Any], Str[Any], Str[Any], Str[Any]]] = deque()

        for s1 in t.start:
            for s2 in t.start:
                if (s1, s2) not in trimmed_product:
                    continue
                visited.add((s1, s2, 0, ()))
                worklist.append((s1, s2, 0, (), (), (), ()))

        while worklist:
            q1, q2, side, buf, x, y1, y2 = worklist.popleft()

            if (q1, q2) in product_final and (side != 0 or buf != ()):
                return (False, (x, y1, y2))

            # Delay cannot exceed the bound: any divergence at product state P
            # can exit to a product_final in at most |trimmed_product| steps,
            # each growing delay by at most M, so the witness is found at
            # delay ≤ |trimmed_product| * M = delay_bound before this fires.
            assert len(buf) <= delay_bound, (
                f"delay {len(buf)} exceeded bound {delay_bound}"
            )

            for a, b1, b2, j1, j2 in product_arcs.get((q1, q2), ()):
                if (j1, j2) not in trimmed_product:
                    continue
                new_side, new_buf = _advance_buf(side, buf, b1, b2)
                key = (j1, j2, new_side, new_buf)
                if key in visited:
                    continue
                visited.add(key)
                na = () if a == EPSILON else (a,)
                nb1 = () if b1 == EPSILON else (b1,)
                nb2 = () if b2 == EPSILON else (b2,)
                worklist.append((j1, j2, new_side, new_buf,
                                 x + na, y1 + nb1, y2 + nb2))

        return (True, None)

    def trim(self) -> FST[A, B]:
        """
        Return a new FST containing only the states and arcs lying on
        some start → stop path.
        """

        trimmed_states = self.reachable() & self.coreachable()

        # ---- collect arcs within trimmed states ----
        trimmed: FST[A, B] = FST(
            start=self.start & trimmed_states,
            stop=self.stop & trimmed_states,
        )
        for q in trimmed_states:
            for x, y, dst in self.arcs(q):
                if dst in trimmed_states:
                    trimmed.add_arc(q, x, y, dst)

        return trimmed

    def push_labels(self) -> FST[A, B]:
        """Push output labels toward initial states to reduce output delay.

        Returns a new FST that realizes the same relation but with output
        produced as early as possible.  This reduces powerset state diversity
        during decomposition by advancing the buffer position sooner.
        """
        t = self.trim()
        if not t.states:
            return t

        # --- 1. Compute output delays via least fixpoint ---
        # d(q) is the longest common prefix of all output strings reachable
        # from q to any final state.  We use None as "unconstrained" (top of
        # lattice) — meaning we haven't yet seen any path from q.

        delay: dict[State, Str[B] | None] = {}
        for q in t.states:
            if q in t.stop:
                delay[q] = ()        # final state: empty path contributes ε
            else:
                delay[q] = None      # unconstrained until resolved

        # Build reverse adjacency so we can propagate backwards
        reverse_adj: defaultdict[State, list[tuple[State, Str[B]]]] = defaultdict(list)
        for q in t.states:
            for x, y, qp in t.arcs(q):
                y_tuple = (y,) if y != EPSILON else ()
                reverse_adj[qp].append((q, y_tuple))

        # Iterative fixpoint
        changed = True
        while changed:
            changed = False
            for q in t.states:
                # d(q) = LCP({concat(y, d(q')) for each arc (q, x, y, q')})
                new_val = None
                for x, y, qp in t.arcs(q):
                    y_tuple = (y,) if y != EPSILON else ()
                    d_qp = delay[qp]
                    if d_qp is None:
                        continue     # unconstrained successor, skip
                    candidate = y_tuple + d_qp
                    if new_val is None:
                        new_val = candidate
                    else:
                        new_val = _lcp_pair(new_val, candidate)
                # Also, if q is final, the empty path contributes ε
                if q in t.stop:
                    if new_val is None:
                        new_val = ()
                    else:
                        new_val = _lcp_pair(new_val, ())

                if new_val is not None and new_val != delay[q]:
                    delay[q] = new_val
                    changed = True

        # After trim(), every state lies on a start→stop path, so the
        # iterative fixpoint always propagates delay from stop states
        # backward through all reachable predecessors.
        for q in t.states:
            assert delay[q] is not None, f"fixpoint failed to resolve delay for state {q} after trim"

        # After fixpoint, all delays are resolved (non-None).
        delay_resolved: dict[State, Str[B]] = {q: d for q, d in delay.items() if d is not None}

        # --- 2. Transform arcs ---
        result: FST[A, B] = FST()
        counter = [0]
        tag = object()   # unique per call so repeated pushes don't collide

        def fresh() -> State:
            counter[0] += 1
            return ('__push__', tag, counter[0])

        for q in t.states:
            for x, y, qp in t.arcs(q):
                y_tuple: Str[B] = (y,) if y != EPSILON else ()
                full_output = y_tuple + delay_resolved[qp]
                new_output = _strip_prefix(delay_resolved[q], full_output)

                if len(new_output) == 0:
                    result.add_arc(q, x, EPSILON, qp)  # pyright: ignore[reportArgumentType]
                elif len(new_output) == 1:
                    result.add_arc(q, x, new_output[0], qp)
                else:
                    # Multi-symbol output: insert ε-input intermediates
                    prev = q
                    for k, sym in enumerate(new_output):
                        if k == 0:
                            nxt = fresh()
                            result.add_arc(prev, x, sym, nxt)
                            prev = nxt
                        elif k < len(new_output) - 1:
                            nxt = fresh()
                            result.add_arc(prev, EPSILON, sym, nxt)  # pyright: ignore[reportArgumentType]
                            prev = nxt
                        else:
                            result.add_arc(prev, EPSILON, sym, qp)  # pyright: ignore[reportArgumentType]

        # --- 3. Handle start states ---
        for s in t.start:
            d_s = delay_resolved[s]
            if len(d_s) == 0:
                result.add_start(s)
            else:
                # Prepend ε-input chain producing d(s)
                prev = fresh()
                result.add_start(prev)
                for k, sym in enumerate(d_s):
                    if k < len(d_s) - 1:
                        nxt = fresh()
                        result.add_arc(prev, EPSILON, sym, nxt)  # pyright: ignore[reportArgumentType]
                        prev = nxt
                    else:
                        result.add_arc(prev, EPSILON, sym, s)  # pyright: ignore[reportArgumentType]

        # --- 4. Final states ---
        for q in t.stop:
            result.add_stop(q)

        return result.trim()

    def reachable(self) -> set[State]:
        """Return the set of states reachable from any start state."""
        reachable: set[State] = set()
        dq: deque[State] = deque(self.start)
        while dq:
            s = dq.popleft()
            if s in reachable:
                continue
            reachable.add(s)
            for _,_,t in self.arcs(s):
                if t not in reachable:
                    dq.append(t)
        return reachable

    def coreachable(self) -> set[State]:
        """Return the set of states from which some stop state is reachable."""
        radj: defaultdict[State, set[State]] = defaultdict(set)
        for q in self.states:
            for _, _, dst in self.arcs(q):
                radj[dst].add(q)
        coreachable: set[State] = set()
        dq: deque[State] = deque(self.stop)
        while dq:
            s = dq.popleft()
            if s in coreachable:
                continue
            coreachable.add(s)
            for t in radj[s]:
                if t not in coreachable:
                    dq.append(t)
        return coreachable

    #___________________________________________________________________________
    #



def _advance_buf(side: int, buf: Str[Any], b1: Any, b2: Any) -> tuple[int, Str[Any]]:
    """Update the output-buffer difference after copy-1 emits b1 and copy-2 emits b2.

    The buffer tracks what one copy has emitted beyond the other as a tuple
    of output labels:
      side=0, buf=() means the two copies are in sync.
      side=1, buf=s  means copy-1 is ahead by suffix s.
      side=2, buf=s  means copy-2 is ahead by suffix s.
    """
    # Build the full "ahead" tuple for each side
    if side == 0:
        ahead1, ahead2 = (), ()
    elif side == 1:
        ahead1, ahead2 = buf, ()
    else:
        ahead1, ahead2 = (), buf

    if b1 != EPSILON:
        ahead1 = ahead1 + (b1,)
    if b2 != EPSILON:
        ahead2 = ahead2 + (b2,)

    # Cancel common prefix
    n = min(len(ahead1), len(ahead2))
    common = 0
    for i in range(n):
        if ahead1[i] != ahead2[i]:
            break
        common = i + 1

    ahead1 = ahead1[common:]
    ahead2 = ahead2[common:]

    if ahead1:
        return (1, ahead1)
    elif ahead2:
        return (2, ahead2)
    else:
        return (0, ())


def _lcp_pair(a: Str[Any], b: Str[Any]) -> Str[Any]:
    """Longest common prefix of two tuples."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return a[:i]
    return a[:n]


def _strip_prefix(prefix: Str[Any], seq: Str[Any]) -> Str[Any]:
    """Remove prefix from seq, returning the remainder as a tuple."""
    assert seq[:len(prefix)] == prefix, f'{prefix} is not a prefix of {seq}'
    return seq[len(prefix):]



def epsilon_filter_fst(Sigma: set[Any]) -> FST[Any, Any]:
    """
    Returns the 3-state epsilon-filtered FST, that is used in to avoid
    epsilon-related ambiguity when composing WFST with epsilons.
    """

    F: FST[Any, Any] = FST()

    F.add_start(0)

    for a in Sigma:
        F.add_arc(0, a, a, 0)
        F.add_arc(1, a, a, 0)
        F.add_arc(2, a, a, 0)

    F.add_arc(0, ε_2, ε_1, 0)
    F.add_arc(0, ε_1, ε_1, 1)
    F.add_arc(0, ε_2, ε_2, 2)

    F.add_arc(1, ε_1, ε_1, 1)
    F.add_arc(2, ε_2, ε_2, 2)

    F.add_stop(0)
    F.add_stop(1)
    F.add_stop(2)

    return F
