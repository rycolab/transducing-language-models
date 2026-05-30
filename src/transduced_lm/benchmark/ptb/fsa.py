"""Finite-state automaton (FSA) data structure with regular-language operations.

Provides the :class:`FSA` class for building and manipulating finite-state
acceptors.  Supports union, concatenation, intersection, difference,
complement, Kleene star, left/right quotient, determinization, minimization,
epsilon removal, and language equivalence testing.  Also includes Graphviz
single-tape acceptors.
"""

from __future__ import annotations

import builtins
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, overload

from .util import Integerizer, State, Str

if TYPE_CHECKING:
    pass

A = TypeVar('A')
B = TypeVar('B')

EPSILON = eps = ''


def dfs(Ps: Iterable[State], arcs: Callable[[State], Iterable[tuple[A, State]]]) -> FSA[A]:
    """Build an FSA by depth-first exploration from seed states.

    Args:
        Ps: Iterable of start states.
        arcs: Callable ``arcs(state) -> iterable of (label, successor)`` pairs.

    Returns:
        An FSA whose states and arcs are the DFS-reachable closure of ``Ps``.
    """
    stack = list(Ps)
    m: FSA[A] = FSA()
    for P in Ps: m.add_start(P)
    while stack:
        P = stack.pop()
        for a, Q in arcs(P):
            if Q not in m.states:
                stack.append(Q)
                m.states.add(Q)
            m.add(P, a, Q)
    return m


def trimmed_fsa(
    start_states: Iterable[State],
    stop_states: Iterable[State],
    get_incoming: Callable[[State], Iterable[tuple[Any, State]]],
) -> FSA[Any]:
    """Build a trimmed FSA by backward BFS from stop states through a
    reverse-arc graph.  All states in the incoming index are assumed
    forward-reachable (guaranteed by the BFS that built them), so backward
    reachability from stops gives exactly the trim (forward ∩ backward
    reachable) set.

    Args:
        start_states: Forward-reachable start states of the FSA.
        stop_states: Accepting states to trace back from.
        get_incoming: ``get_incoming(state)`` returns ``(symbol, predecessor)``
            pairs — the reverse-arc index.
    """
    stop_states = set(stop_states)
    if not stop_states:
        return FSA()
    backward_reachable: set[State] = set()
    worklist = deque(stop_states)
    while worklist:
        state = worklist.popleft()
        if state in backward_reachable:
            continue
        backward_reachable.add(state)
        for _, pred in get_incoming(state):
            if pred not in backward_reachable:
                worklist.append(pred)
    arcs = [
        (pred, x, state)
        for state in backward_reachable
        for x, pred in get_incoming(state)
        if pred in backward_reachable
    ]
    return FSA(
        start={s for s in start_states if s in backward_reachable},
        arcs=arcs,
        stop=stop_states,
    )


class frozenset(builtins.frozenset):  # type: ignore[type-arg]
    "Same as frozenset, but with a nicer printing method."
    def __repr__(self) -> str:
        return '{%s}' % (','.join(str(x) for x in self))  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]


class FSA(Generic[A]):
    """Finite-state automaton (acceptor) over a symbolic alphabet.

    An FSA is a directed graph whose arcs carry single labels.  A path
    from a start state to a stop state spells out a string; the set of
    all such strings is the FSA's language.

    Supports regular-language operations: union (``+`` / ``|``),
    concatenation (``*``), intersection (``&``), difference (``-``),
    complement (``invert``), Kleene star (``star``), left/right
    quotient (``//`` / ``/``), determinization (``det``),
    minimization (``min``), and language equivalence (``equal``).

    Attributes:
        states: Set of all states.
        start: Set of initial states.
        stop: Set of final (accepting) states.
        syms: Set of arc labels observed (excludes EPSILON).
    """

    one: ClassVar[FSA[Any]]
    zero: ClassVar[FSA[Any]]

    def __init__(
        self,
        start: Iterable[State] = (),
        arcs: Iterable[tuple[State, A, State]] = (),
        stop: Iterable[State] = (),
    ) -> None:
        """Create an FSA, optionally populating it from iterables.

        Args:
            start: Iterable of initial states.
            arcs: Iterable of ``(src, label, dst)`` tuples.
            stop: Iterable of final (accepting) states.
        """
        self.states: set[State] = set()
        self.start: set[State] = set()
        self.stop: set[State] = set()
        self.syms: set[A] = set()
        self.edges: defaultdict[State, defaultdict[Any, set[State]]] = defaultdict(lambda: defaultdict(set))
        # use the official methods for the constructor's initialization
        for i in start: self.add_start(i)
        for i in stop: self.add_stop(i)
        for i,a,j in arcs: self.add_arc(i,a,j)

    def materialize(self) -> FSA[A]:
        # the FSA is already materialized
        return self


    def __hash__(self) -> int:
        return hash((frozenset(self.states), frozenset(self.start),
                     frozenset(self.stop), frozenset(self.arcs())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FSA):
            return NotImplemented
        return (self.states == other.states
                and self.start == other.start
                and self.stop == other.stop
                and set(self.arcs()) == set(other.arcs()))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    def __repr__(self) -> str:
        x = ['{']
        for s in self.states:
            ss = f'{s}'
            if s in self.start:
                ss = f'^{ss}'
            if s in self.stop:
                ss = f'{ss}$'
            x.append(f'  {ss}:')
            for a, t in self.arcs(s):
                x.append(f'    {a} -> {t}')
        x.append('}')
        return '\n'.join(x)



    def D(self, x: A) -> FSA[A]:
        """Left derivative: the language of strings ``y`` such that ``x·y`` is accepted."""
        e = self.epsremove()
        m: FSA[A] = FSA(start = set(e.start), stop = set(e.stop))
        for i,a,j in e.arcs():
            if i in e.start and a == x:
                m._add_eps(i, j)
            else:
                m.add(i,a,j)
        return m

    def _add_eps(self, i: State, j: State) -> None:
        """Add an epsilon arc from ``i`` to ``j`` (internal helper)."""
        self.edges[i][EPSILON].add(j)
        self.states.add(i)
        self.states.add(j)

    def add(self, i: State, a: A, j: State) -> FSA[A]:
        """Add arc from state ``i`` to state ``j`` with label ``a``. Creates states implicitly."""
        self.edges[i][a].add(j)
        self.states.add(i)
        self.states.add(j)
        if a != EPSILON:
            self.syms.add(a)
        return self

    add_arc = add

    def add_start(self, i: State) -> FSA[A]:
        """Mark state ``i`` as an initial state (creates it if needed)."""
        self.start.add(i)
        self.states.add(i)
        return self

    def add_stop(self, i: State) -> FSA[A]:
        """Mark state ``i`` as a final (accepting) state (creates it if needed)."""
        self.stop.add(i)
        self.states.add(i)
        return self

    def is_final(self, i: State) -> bool:
        """Return True if state ``i`` is a final (accepting) state."""
        return i in self.stop

    @overload
    def arcs(self) -> Iterator[tuple[State, A, State]]: ...
    @overload
    def arcs(self, i: State) -> Iterator[tuple[A, State]]: ...
    @overload
    def arcs(self, i: State, a: A) -> Iterator[State]: ...

    def arcs(self, i: State | None = None, a: A | None = None) -> Iterator[Any]:
        """Iterate over arcs.

        - ``arcs()`` → yields all ``(i, a, j)`` triples.
        - ``arcs(i)`` → yields ``(a, j)`` pairs for arcs from state ``i``.
        - ``arcs(i, a)`` → yields destination states ``j`` for arcs ``i --a--> j``.
        """
        if i is None and a is None:

            for i in self.edges:
                for a in self.edges[i]:
                    for j in self.edges[i][a]:
                        yield (i,a,j)

        elif i is not None and a is None:

            for a in self.edges[i]:
                for j in self.edges[i][a]:
                    yield (a,j)

        elif i is not None and a is not None:

            for j in self.edges[i][a]:
                yield j

        else:
            raise NotImplementedError()

    def arcs_x(self, i: State, x: A) -> set[State]:
        """Return the set of destination states for arcs ``i --x--> *``."""
        return self.edges[i][x]

    def reverse(self) -> FSA[A]:
        """Return the reversal: arcs are flipped, start and stop states are swapped."""
        m: FSA[A] = FSA()
        for i in self.start:
            m.add_stop(i)
        for i in self.stop:
            m.add_start(i)
        for i, a, j in self.arcs():
            m.add(j, a, i)     # pylint: disable=W1114
        return m

    def _accessible(self, start: set[State]) -> set[State]:
        return dfs(start, self.arcs).states

    def accessible(self) -> set[State]:
        """Return the set of states reachable from any start state."""
        return self._accessible(self.start)

    def trim(self) -> FSA[A]:
        """Return a copy keeping only states on some start-to-stop path."""
        c = self.accessible() & self.reverse().accessible()
        m: FSA[A] = FSA()
        for i in self.start & c:
            m.add_start(i)
        for i in self.stop & c:
            m.add_stop(i)
        for i,a,j in self.arcs():
            if i in c and j in c:
                m.add(i,a,j)
        return m

    def renumber(self) -> FSA[A]:
        """Return a copy with states relabeled as consecutive integers."""
        return self.rename(Integerizer())  # pyright: ignore[reportUnknownArgumentType]

    def rename(self, f: Callable[[State], State]) -> FSA[A]:
        """Return a new FSA with states relabeled by ``f(state)``.

        If ``f`` is not injective, distinct states may merge.
        """
        m: FSA[A] = FSA()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i, a, j in self.arcs():
            m.add(f(i), a, f(j))
        return m

    def map_labels(self, f: Callable[[A], B]) -> FSA[B]:
        "Transform arc labels by applying f to each label."
        m: FSA[B] = FSA()
        for i in self.start:
            m.add_start(i)
        for i in self.stop:
            m.add_stop(i)
        for i, a, j in self.arcs():
            m.add(i, f(a), j)
        return m

    def rename_apart(self, other: FSA[A]) -> tuple[FSA[A], FSA[A]]:
        """Rename states of ``self`` and ``other`` so their state sets are disjoint."""
        f: Integerizer[State] = Integerizer()
        self = self.rename(lambda i: f((0, i)))
        other = other.rename(lambda i: f((1, i)))
        assert self.states.isdisjoint(other.states)
        return (self, other)

    def __mul__(self, other: FSA[A]) -> FSA[A]:
        """Concatenation: ``self * other`` accepts strings ``xy`` where ``x in self`` and ``y in other``."""
        self, other = self.rename_apart(other)
        m: FSA[A] = FSA(
            start = self.start,
            stop = other.stop,
        )
        for i in self.stop:
            for j in other.start:
                m._add_eps(i, j)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def __add__(self, other: FSA[A]) -> FSA[A]:
        """Union: ``self + other`` accepts strings in either language."""
        m: FSA[A] = FSA()
        [self, other] = self.rename_apart(other)
        m.start = self.start | other.start
        m.stop = self.stop | other.stop
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def p(self) -> FSA[A]:
        """Kleene plus: one or more repetitions of ``self``."""
        m: FSA[A] = FSA()
        m.start = set(self.start)
        m.stop = set(self.stop)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i in self.stop:
            m.add_stop(i)
            for j in self.start:
                m._add_eps(i, j)
        return m

    def star(self) -> FSA[A]:
        """Kleene star: zero or more repetitions of ``self``."""
        return one + self.p()

    def epsremove(self) -> FSA[A]:
        """Return an equivalent FSA with all epsilon transitions removed."""
        eps_m: FSA[A] = FSA()
        for i,a,j in self.arcs():
            if a == eps:
                eps_m._add_eps(i, j)

        @lru_cache
        def eps_accessible(i: State) -> set[State]:
            return eps_m._accessible({i})

        m: FSA[A] = FSA()

        for i,a,j in self.arcs():
            if a == eps: continue
            m.add(i, a, j)
            for k in eps_accessible(j):
                m.add(i, a, k)

        for i in self.start:
            m.add_start(i)
            for k in eps_accessible(i):
                m.add_start(k)

        for i in self.stop:
            m.add_stop(i)

        return m

    def det(self) -> FSA[A]:
        """Determinize via the powerset (subset) construction. Returns a DFA."""
        self = self.epsremove()

        def powerarcs(Q: State) -> Iterator[tuple[A, State]]:
            for a in self.syms:
                yield a, frozenset(j for i in Q for j in self.edges[i][a])

        m = dfs([frozenset(self.start)], powerarcs)

        for powerstate in m.states:
            if powerstate & self.stop:
                m.add_stop(powerstate)

        return m


    def min_fast(self) -> FSA[A]:
        """Minimize via Hopcroft's partition-refinement algorithm.

        Determinizes first, then iteratively splits equivalence classes.
        O(n log n) in theory.  See ``min_faster`` for an optimized variant.
        """
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv: defaultdict[tuple[State, A], set[State]] = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.states - final

        P: list[set[State]] = [final, nonfinal]
        W: list[set[State]] = [final, nonfinal]

        while W:
            S = W.pop()
            for a in self.syms:
                X = {i for j in S for i in inv[j,a]}
                R: list[set[State]] = []
                for Y in P:
                    if X.isdisjoint(Y) or X >= Y:
                        R.append(Y)
                    else:
                        YX = Y & X
                        Y_X = Y - X
                        R.append(YX)
                        R.append(Y_X)
                        W.append(YX if len(YX) < len(Y_X) else Y_X)
                P = R

        # create new equivalence classes of states
        minstates: dict[State, int] = {}
        for i, qs in enumerate(P):
            for q in qs:
                minstates[q] = i #minstate

        return self.rename(lambda i: minstates[i]).trim()

    def min_faster(self) -> FSA[A]:
        """Optimized Hopcroft minimization with a ``find`` index for O(1) block lookup.

        Same algorithm as ``min_fast`` but avoids scanning all blocks on each
        refinement step.  This is the default (aliased as ``min``).
        """
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv: defaultdict[tuple[State, A], set[State]] = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.states - final

        P: list[set[State]] = [final, nonfinal]
        W: list[set[State]] = [final, nonfinal]

        find: dict[State, int] = {i: block for block, elements in enumerate(P) for i in elements}

        while W:

            S = W.pop()
            for a in self.syms:

                # Group pre-images by their current block; this lets us
                # replace the O(|Y|) superset check `X >= Y` with an
                # O(1) length comparison.
                block_members: defaultdict[int, set[State]] = defaultdict(set)
                for j in S:
                    for i in inv[j, a]:
                        block_members[find[i]].add(i)

                for block, YX in block_members.items():
                    Y = P[block]

                    if len(YX) == len(Y): continue

                    Y_X = Y - YX

                    # we will replace block with the intersection case (no
                    # need to update `find` index for YX elements)
                    P[block] = YX

                    new_block = len(P)
                    for i in Y_X:
                        find[i] = new_block

                    P.append(Y_X)
                    W.append(YX if len(YX) < len(Y_X) else Y_X)

        return self.rename(lambda i: find[i]).trim()

    min = min_faster

    def equal(self, other: FSA[A] | frozenset | list[Any] | set[Any] | tuple[Any, ...]) -> bool:
        """Test language equivalence via minimal-DFA isomorphism."""
        if isinstance(other, (frozenset, list, set, tuple)):
            other = FSA.from_strings(other)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return self.min()._dfa_isomorphism(other.min())

    def _dfa_isomorphism(self, other: FSA[A]) -> bool:
        "Find isomorphism between DFAs (if one exists)."

        # Requires that self and other are minimal DFAs

        # Theorem. If `self` and `other` are graphs with out-degree at most 1, then
        # the DFA works to determine whether G and H are isomorphic

        # A deterministic machine has exactly one start state

        # Two minimized DFAs are input
        # If the number of states is differs, these machines cannot be isomorphic
        if len(self.states) != len(other.states): return False
        if len(self.start) == 0: return len(other.start) == 0

        assert len(self.start) == 1 and len(other.start) == 1

        [p] = self.start
        [q] = other.start

        stack = [(p, q)]
        iso: dict[State, State] = {p: q}

        syms = self.syms | other.syms

        done: set[tuple[State, State]] = set()
        while stack:
            (p, q) = stack.pop()
            done.add((p,q))
            for a in syms:

                # presences of the arc has to be the same
                if (a in self.edges[p]) != (a in other.edges[q]):
                    return False

                if a not in self.edges[p]:
                    continue

                # machines are assumed deterministic
                [r] = self.edges[p][a]
                [s] = other.edges[q][a]

                if r in iso and iso[r] != s:
                    return False

                iso[r] = s
                if (r,s) not in done:
                    stack.append((r,s))

        return self.rename(iso.get) == other

    def __and__(self, other: FSA[A]) -> FSA[A]:
        """Intersection: ``self & other`` accepts strings in both languages."""

        self = self.epsremove().renumber()
        other = other.epsremove().renumber()

        def product_arcs(Q: State) -> Iterator[tuple[A, State]]:
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield a, (j1,j2)

        m = dfs({(q1, q2) for q1 in self.start for q2 in other.start},
                product_arcs)

        # final states
        for q1 in self.stop:
            for q2 in other.stop:
                m.add_stop((q1, q2))

        return m

    def add_sink(self, syms: Iterable[A]) -> FSA[A]:
        """Make the FSA complete over ``syms`` by adding a non-accepting sink state for missing transitions."""

        syms_set = set(syms)

        self = self.renumber()

        sink = len(self.states)
        for a in syms_set:
            self.add(sink, a, sink)

        for q in self.states:
            if q == sink: continue
            for a in syms_set - set(self.edges[q]):
                if a == eps: continue  # ignore epsilon
                self.add(q, a, sink)

        return self

    def __sub__(self, other: FSA[A]) -> FSA[A]:
        """Set difference: ``self - other`` accepts strings in ``self`` but not in ``other``."""
        return self & other.invert(self.syms | other.syms)

    __or__ = __add__

    def __xor__(self, other: FSA[A]) -> FSA[A]:
        "Symmetric difference"
        return (self | other) - (self & other)

    def invert(self, syms: Iterable[A]) -> FSA[A]:
        """Complement over alphabet ``syms``: accepts all strings not in the language."""

        self = self.det().add_sink(syms)

        m: FSA[A] = FSA()

        for i in self.states:
            for a, j in self.arcs(i):
                m.add(i, a, j)

        for q in self.start:
            m.add_start(q)

        for q in self.states - self.stop:
            m.add_stop(q)

        return m

    def __floordiv__(self, other: FSA[A]) -> FSA[A]:
        "left quotient self//other ≐ {y | ∃x ∈ other: x⋅y ∈ self}"

        self = self.epsremove()
        other = other.epsremove()

        # quotient arcs are very similar to product arcs except that the common
        # string is "erased" in the new machine.
        def quotient_arcs(Q: State) -> Iterator[tuple[str, State]]:
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield EPSILON, (j1, j2)

        m: FSA[A] = dfs(
            {(q1, q2) for q1 in self.start for q2 in other.start},
            quotient_arcs,  # type: ignore[arg-type]
        )

        # If we have managed to reach a final state of q2 then we can move into
        # the post-prefix set of states
        for (q1,q2) in set(m.states):
            if q2 in other.stop:
                m._add_eps((q1, q2), (q1,))

        # business as usual
        for q1 in self.states:
            for a, j1 in self.arcs(q1):
                m.add((q1,), a, (j1,))
        for q1 in self.stop:
            m.add_stop((q1,))

        return m

    def __truediv__(self, other: FSA[A]) -> FSA[A]:
        "right quotient self/other ≐ {x | ∃y ∈ other: x⋅y ∈ self}"
        return (self.reverse() // other.reverse()).reverse()   # reduce to left quotient on reversed languages

    def __lt__(self, other: FSA[A]) -> bool:
        "self ⊂ other"
        if self.equal(other): return False
        return (self & other).equal(self)

    def __le__(self, other: FSA[A]) -> bool:
        "self ⊆ other"
        return (self & other).equal(self)

    @classmethod
    def lift(cls, x: A) -> FSA[A]:
        """Build a single-arc FSA accepting exactly the one-symbol string ``x``."""
        m: FSA[A] = cls()
        m.add_start(0); m.add_stop(1); m.add(0,x,1)
        return m

    @classmethod
    def from_string(cls, xs: Sequence[A]) -> FSA[A]:
        """Build a linear FSA accepting exactly the string ``xs``."""
        m: FSA[A] = cls()
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add(xs[:i], xs[i], xs[:i+1])
        m.add_stop(xs)
        return m

    @classmethod
    def from_strings(cls, Xs: Iterable[Sequence[A]]) -> FSA[A]:
        """Build an FSA accepting the union of the given strings (trie structure)."""
        m: FSA[A] = cls()
        for xs in Xs:
            m.add_start(xs[:0])
            for i in range(len(xs)):
                m.add(xs[:i], xs[i], xs[:i+1])
            m.add_stop(xs)
        return m

    def _eps_closure(self, states: set[State]) -> set[State]:
        """Expand *states* by following all epsilon transitions."""
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for t in self.edges[s][EPSILON]:
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
        return closure

    def run(self, xs: Iterable[A]) -> set[State]:
        """Run string ``xs`` from start states, returning the set of reached states."""
        states = self._eps_closure(set(self.start))
        for x in xs:
            states = self._eps_closure({t for s in states for t in self.edges[s][x]})
            if not states:
                break
        return states

    def __contains__(self, xs: Iterable[A]) -> bool:
        """Test if string ``xs`` is in the language: ``xs in fsa``."""
        return bool(self.run(xs) & self.stop)

    def merge(self, S: set[State], name: State | None = None) -> FSA[A]:
        """Merge all states in ``S`` into a single state (default name: ``min(S)``)."""
        if name is None: name = min(S)
        def f(s: State) -> State:
            return name if s in S else s
        m: FSA[A] = FSA()
        for x in self.start:
            m.add_start(f(x))
        for x,a,y in self.arcs():
            m.add(f(x),a,f(y))
        for x in self.stop:
            m.add_stop(f(x))
        return m

    @staticmethod
    def universal(alphabet: Iterable[A]) -> FSA[A]:
        """Build a single-state FSA accepting all strings over ``alphabet`` (Σ*)."""
        u: FSA[A] = FSA()
        u.add_start(0)
        for a in alphabet:
            u.add(0, a, 0)
        u.add_stop(0)
        return u

    def language(self, max_length: int | None = None) -> Iterator[Str[A]]:
        "Enumerate strings in the language of this FSA."
        worklist: deque[tuple[State, Str[A]]] = deque()
        worklist.extend([(i, ()) for i in self.start])
        while worklist:
            (i, x) = worklist.popleft()
            if i in self.stop:
                yield x
            if max_length is not None and len(x) >= max_length:
                continue
            for a, j in self.arcs(i):
                if a == EPSILON:
                    worklist.append((j, x))
                else:
                    worklist.append((j, x + (a,)))


one: FSA[Any] = FSA()
one.add_start(0); one.add_stop(0)
FSA.one = one

zero: FSA[Any] = FSA()
FSA.zero = zero
