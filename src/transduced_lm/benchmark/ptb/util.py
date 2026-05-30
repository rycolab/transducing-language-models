"""Shared utilities: Integerizer, log-space types, caching, memory/time limits.

Key public classes:

- :class:`Integerizer` — bijection between objects and contiguous ints.
- :class:`LogVector` — mutable sparse log-space accumulator (``logaddexp``).
- :class:`LogDistr` — immutable normalized log-probability distribution.
- :class:`memoize` — simple function-result cache (decorator).

Key public functions:

- :func:`logsumexp` — numerically stable log-sum-exp.
- :func:`set_memory_limit` / :func:`memory_limit` — RLIMIT_AS guards.
- :func:`timelimit` — wall-clock timeout via SIGALRM.
"""

from __future__ import annotations

import resource
import signal
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from functools import partial
from types import FrameType
from typing import Any, Generic, TypeAlias, TypeVar, overload

import math

import numpy as np
import numpy.typing as npt
import torch

_T = TypeVar('_T')
Str = tuple[_T, ...]
"""Type alias for strings (sequences of symbols) in automata theory."""

State: TypeAlias = Any
"""Type alias for automaton states.

States can be any hashable type (int, tuple, frozenset, etc.) and are used
heterogeneously across FSA/FST operations (product states are tuples,
powerset states are frozensets, renumbered states are ints).
"""


#_______________________________________________________________________________
# Integerizer (formerly from arsenal)

K = TypeVar('K', bound=Hashable)


class Integerizer(Generic[K]):
    """Maintain a perfect hash (bijection) between keys and contiguous ints."""

    def __init__(self, data: list[K] | None = None) -> None:
        """Create an Integerizer, optionally adding initial keys from *data*."""
        self._map: dict[K, int] = {}
        self._list: list[K] = []
        self._frozen = False
        if data: self(data)

    def _add(self, k: K) -> int:
        """Return the integer for *k*, assigning a new one if unseen."""
        try:
            return self._map[k]
        except KeyError as exc:
            if self._frozen:
                raise ValueError(f'Alphabet is frozen. Key "{k}" not found.') from exc
            x = self._map[k] = len(self._list)
            self._list.append(k)
            return x

    def __contains__(self, k: object) -> bool:
        """Return True if *k* has been assigned an integer."""
        return k in self._map

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in insertion order."""
        return iter(self._list)

    def __len__(self) -> int:
        """Return the number of registered keys."""
        return len(self._list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integerizer):
            return NotImplemented
        return self._list == other._list  # pyright: ignore[reportUnknownMemberType]

    def __repr__(self) -> str:
        return f'Integerizer(size={len(self)}, frozen={self._frozen})'

    @overload
    def __getitem__(self, i: list[int]) -> list[K]: ...
    @overload
    def __getitem__(self, i: int) -> K: ...
    def __getitem__(self, i: int | list[int]) -> K | list[K]:
        """Decode: integer(s) -> key(s). Accepts an int or list of ints."""
        if isinstance(i, list):
            return [self._list[ii] for ii in i]
        return self._list[i]

    @overload
    def __call__(self, k: list[K]) -> list[int]: ...
    @overload
    def __call__(self, k: K) -> int: ...
    def __call__(self, k: K | list[K]) -> int | list[int]:
        """Encode: key(s) -> integer(s). Accepts a key or list of keys."""
        if isinstance(k, list):
            return [self._add(kk) for kk in k]
        return self._add(k)


    def keys(self) -> list[K]:
        """Return the list of keys in insertion order."""
        return self._list

    def items(self) -> Any:
        """Return (key, int) pairs."""
        return self._map.items()



#_______________________________________________________________________________
# colors (formerly from arsenal)

class _AnsiStyle:
    """ANSI escape code wrapper supporting ``style % string`` formatting."""
    def __init__(self, code: str) -> None:
        self._code = code
    def __mod__(self, text: object) -> str:
        return f'\033[{self._code}m{text}\033[0m'

class _LightColors:
    red = _AnsiStyle('1;31')
    green = _AnsiStyle('1;32')

class _DarkColors:
    white = _AnsiStyle('2;37')

class colors:
    light = _LightColors()
    dark = _DarkColors()
    @staticmethod
    def mark(ok: bool) -> str:
        return '\033[0;32m\u2714\033[0m' if ok else '\033[2;31m\u2718\033[0m'


#_______________________________________________________________________________
# memoize (formerly from arsenal.cache)

class memoize:
    """Cache a function's return value to avoid recalculation."""
    def __init__(self, func: Any) -> None:
        self.func = func
        self.cache: dict[Any, Any] = {}
        try:
            self.__name__: str = func.__name__
            self.__doc__ = func.__doc__
        except AttributeError:
            pass

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None: return self.func
        return partial(self, obj)

    def __call__(self, *args: Any) -> Any:
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self) -> str:
        return f'<memoize({self.func!r})>'


#_______________________________________________________________________________
# timelimit (formerly from arsenal)

class Timeout(Exception):
    pass

@contextmanager
def timelimit(seconds: float | None, sig: signal.Signals = signal.SIGALRM) -> Iterator[None]:
    """Context manager that raises Timeout after `seconds`.

    Saves and restores the previous signal handler. Clears the timer
    on both normal exit and exception paths.
    """
    if seconds is None:
        yield
        return
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        raise Timeout(f'Call took longer than {seconds} seconds.')
    prev = signal.signal(sig, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(sig, prev)


@contextmanager
def timeit(name: str, fmt: str = '{name} ({htime})', header: str | None = None) -> Iterator[None]:
    """Context manager that prints elapsed wall-clock time."""
    import sys
    from time import time
    if header is not None:
        print(header)
    b4 = time()
    yield
    sec = time() - b4
    ht = f'{sec:.4f} sec' if sec < 60 else f'{sec/60:.1f} min'
    print(fmt.format(name=name, htime=ht, sec=sec), file=sys.stderr)


#_______________________________________________________________________________
# memory limits

def set_memory_limit(gb: float | None) -> None:
    """Set process virtual address space limit in gigabytes (RLIMIT_AS).

    No-op if gb is None or non-positive.
    """
    if gb is not None and gb > 0:
        limit = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


@contextmanager
def memory_limit(gb: float | None) -> Iterator[None]:
    """Context manager that sets a memory limit and restores the original on exit.

    Saves the original RLIMIT_AS soft/hard limits, sets a new soft limit
    (preserving the hard limit), and restores on exit.
    """
    if gb is None or gb <= 0:
        yield
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = int(gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


#_______________________________________________________________________________
# logsumexp

def logsumexp(arr) -> float:
    """Numerically stable log-sum-exp over an array of log values."""
    a = torch.as_tensor(arr, dtype=torch.float64)
    mask = a > float('-inf')
    if not mask.any():
        return float('-inf')
    return float(torch.logsumexp(a[mask], dim=0))


def log1mexp(x: float) -> float:
    """Compute ``log(1 - exp(x))`` for *x* < 0, numerically stable.

    Uses the two-branch formula from Mächler (2012):
    - *x* < -log(2): ``log1p(-exp(x))`` (exp(x) < 0.5, no cancellation)
    - -log(2) ≤ *x* < 0: ``log(-expm1(x))`` (expm1 accurate near 0)

    Returns -inf when *x* = 0 (i.e., exp(x) = 1).
    """
    if x >= 0:
        return _NEG_INF
    if x < -0.6931471805599453:   # -log(2)
        return math.log1p(-math.exp(x))
    return math.log(-math.expm1(x))


#_______________________________________________________________________________
# Sparse log-space mappings: LogVector (mutable) and LogDistr (immutable)

_NEG_INF = float('-inf')

K = TypeVar('K', bound=Hashable)


class _SparseLogMap(dict[K, float]):
    """Dict subclass for sparse mappings in log-space (keys -> log-values).

    Missing keys return ``-inf`` (via ``__missing__``).
    """

    def __missing__(self, key: K) -> float:
        return _NEG_INF

    def materialize(self, top: int | None = None) -> dict[K, float]:
        """Return a dict of ``{key: value}`` sorted by descending value.

        If ``top`` is given, return only the top-k entries.
        """
        items = sorted(self.items(), key=lambda kv: kv[1], reverse=True)
        if top is not None:
            items = items[:int(top)]
        return dict(items)

    def top(self, k: int) -> dict[K, float]:
        """Return a dict of the top-k entries by value."""
        return self.materialize(top=k)

    def argmax(self) -> K:
        """Return the key with the highest value."""
        return max(self, key=self.__getitem__)

    def __repr__(self) -> str:
        name = type(self).__name__
        n = len(self)
        if n <= 5:
            inner = ', '.join(f'{k!r}: {v:.4f}' for k, v in self.items())
            return f'{name}({{{inner}}})'
        return f'{name}(n={n})'



class LogVector(_SparseLogMap[K]):
    """Mutable accumulator for sparse log-nonneg-real vectors.

    Replaces the ``defaultdict(lambda: -np.inf)`` + ``logaddexp`` pattern.
    """

    def logaddexp(self, key: K, value: float) -> None:
        """Accumulate: ``self[key] = logaddexp(self[key], value)``."""
        prev = self.get(key)
        if prev is None:
            self[key] = value
        else:
            # Inline logaddexp to avoid tensor creation overhead.
            # logaddexp(a, b) = max(a,b) + log1p(exp(-|a-b|))
            a, b = prev, value
            if a < b:
                a, b = b, a
            if b == _NEG_INF:
                self[key] = a
            else:
                self[key] = a + math.log1p(math.exp(b - a))

    def normalize(self) -> LogDistr[K]:
        """Return a ``LogDistr`` by subtracting ``logsumexp`` from each entry."""
        Z = logsumexp(list(self.values()))
        return LogDistr({k: float(v - Z) for k, v in self.items()})


class LogDistr(_SparseLogMap[K]):
    """Immutable normalized distribution in log-space.

    Supports ``sample()`` but not mutation.

    Optionally tensor-backed: use ``LogDistr.from_tensor(keys, values)``
    to create a distribution backed by a ``torch.Tensor``.  When
    ``_p`` is set, batch operations can extract values via integer
    indexing instead of dict lookups.
    """

    _p: torch.Tensor | None
    _key_to_idx: dict[K, int] | None

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_p'):
            self._p = None
            self._key_to_idx = None

    @classmethod
    def from_tensor(cls, keys: list[K], values: torch.Tensor) -> LogDistr[K]:
        """Create a tensor-backed LogDistr from parallel keys and values.

        Args:
            keys: list of K keys (length V)
            values: 1-D tensor of log-probabilities (length V)

        The resulting LogDistr behaves like a normal dict but also exposes
        ``_p`` (the tensor) and ``_key_to_idx`` (key → position mapping)
        for fast batch extraction.
        """
        d = cls(zip(keys, values.tolist()))
        d._p = values
        d._key_to_idx = {k: i for i, k in enumerate(keys)}
        return d

    def sample(self) -> K:
        """Sample a key proportional to ``exp(value)``."""
        toks = list(self.keys())
        logps = torch.tensor(list(self.values()), dtype=torch.float64)
        logps -= logps.max()
        probs = torch.exp(logps)
        probs /= probs.sum()
        return toks[int(torch.multinomial(probs, 1).item())]


#_______________________________________________________________________________
# sample (formerly from arsenal.maths)

def validate_target(target: Any, target_alphabet: set[Any]) -> tuple[Any, ...]:
    """Coerce *target* to a tuple and check for out-of-vocabulary symbols.

    Returns the tuple-ified target on success; raises ``ValueError`` listing
    OOV symbols otherwise.  Every decomposition algorithm repeats the same
    three-line pattern — centralising it here keeps them DRY.
    """
    target = tuple(target)
    oov = set(target) - target_alphabet
    if oov:
        raise ValueError(f"Out of vocabulary target symbols: {oov}")
    return target


def sample(w: npt.ArrayLike, size: int | None = None, u: npt.ArrayLike | None = None) -> Any:
    """Draw samples from an (unnormalized) discrete distribution via inverse CDF."""
    c = np.cumsum(w)
    if u is None:
        u = np.random.uniform(0, 1, size=size)
    return c.searchsorted(u * c[-1])  # pyright: ignore[reportUnknownVariableType]
