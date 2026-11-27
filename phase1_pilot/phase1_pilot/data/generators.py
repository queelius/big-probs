"""3-regime stratified dataset generators for the Phase 1 Latent-CoT RL Pilot.

Regimes (per Track 1 §5.2):
  - Regime 1 (algorithmic): Controllable multi-hop DAG planning (ProsQA-style),
    exact Python stateful register execution traces, and modular variable binding
    chains with strictly controllable hop depth d in {2, 4} (ID) vs {6, 8} (OOD).
  - Regime 2 (natural): Deterministic structured synthetic math-prose/code stream
    for CPU testing + HuggingFace streaming loader for the 40% OpenWebMath /
    40% The Stack v2 / 20% FineWeb-Edu H100 mixture.
  - Regime 3 (aleatoric): Arbitrary random entity-attribute / proper-noun cloze
    mappings with mathematically irreducible entropy (negative control).
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from phase1_pilot.config import DataConfig
from phase1_pilot.data.tokenizer import PilotTokenizer


@dataclass(frozen=True)
class Sample:
    """Raw text sample across all 3 regimes."""

    prompt: str
    target: str
    full_text: str
    regime: str
    task: str
    depth: int
    answer_span_start: int


@dataclass(frozen=True)
class TokenizedSample:
    """Tokenized sample aligned for supervised NTP, latent BPTT, and SSL-as-RL."""

    input_ids: list[int]
    prompt_len: int
    target_ids: list[int]
    regime: str
    task: str
    depth: int


# ---------------------------------------------------------------------------
# Regime 1: Controllable Algorithmic & Compositional Generators
# ---------------------------------------------------------------------------


def _bfs_shortest_distances(
    edges: Sequence[tuple[str, str]],
    start: str,
) -> dict[str, int]:
    """Exact BFS shortest-path distance from `start` in a directed graph."""
    adj: dict[str, list[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    dist: dict[str, int] = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        cur = queue.popleft()
        cur_d = dist[cur]
        for nxt in adj.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = cur_d + 1
                queue.append(nxt)
    return dist


def generate_dag_planning(
    n: int,
    depths: Sequence[int] = (2, 4),
    seed: int = 42,
    exclude_signatures: set[str] | None = None,
) -> list[Sample]:
    """Generate ProsQA-style DAG multi-hop reachability queries of exact depth `d`.

    Each DAG contains:
      - One true chain `v_0 -> v_1 -> ... -> v_d` of exact length `d`
      - Two parallel distractor chains of identical length `d` terminating in
        candidate leaves `w_d` and `z_d` (unreachable from `v_0`)
      - One decoy branch off `v_0` (or an early node) ending in a non-candidate
        dead-end node so simple traversal must branch/search
      - Shuffled edge order and shuffled candidate list `{v_d, w_d, z_d}`

    Exact BFS verification asserts that ONLY `v_d` among the 3 candidates is
    reachable from `v_0`, at shortest-path distance strictly equal to `d`.
    """
    if n <= 0:
        return []
    if not depths or any(d < 1 for d in depths):
        raise ValueError(f"depths must be non-empty positive ints, got {depths}")

    rng = random.Random(seed)
    excluded = exclude_signatures if exclude_signatures is not None else set()
    local_seen: set[str] = set()
    samples: list[Sample] = []

    # Pool of 32 compact node tokens (N0..N31) -> fits d=8 cleanly: 3*(8+1)+2 = 29 <= 32.
    node_pool = [f"N{i}" for i in range(32)]

    attempts = 0
    max_attempts = max(n * 25, 500)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        d = int(depths[len(samples) % len(depths)])
        needed = 3 * (d + 1) + 2
        if needed > len(node_pool):
            pool = [f"N{i}" for i in range(needed + 4)]
        else:
            pool = node_pool

        nodes = rng.sample(pool, needed)
        true_chain = nodes[0 : d + 1]
        dist_chain_1 = nodes[d + 1 : 2 * (d + 1)]
        dist_chain_2 = nodes[2 * (d + 1) : 3 * (d + 1)]
        decoy_nodes = nodes[3 * (d + 1) :]

        edges: list[tuple[str, str]] = []
        for chain in (true_chain, dist_chain_1, dist_chain_2):
            for i in range(d):
                edges.append((chain[i], chain[i + 1]))

        # Add a branching decoy twig off an early node on the true chain
        # (never connecting to any candidate leaf).
        branch_idx = rng.randint(0, max(0, d - 2))
        edges.append((true_chain[branch_idx], decoy_nodes[0]))
        edges.append((decoy_nodes[0], decoy_nodes[1]))

        start = true_chain[0]
        target = true_chain[-1]
        candidates = [target, dist_chain_1[-1], dist_chain_2[-1]]
        rng.shuffle(candidates)
        rng.shuffle(edges)

        # Canonical structural signature for strict train/eval deduplication.
        sig = f"dag|{d}|{start}|{','.join(sorted(candidates))}|{';'.join(sorted(f'{u}>{v}' for u, v in edges))}"
        if sig in excluded or sig in local_seen:
            continue

        # Exact BFS ground-truth verification.
        dists = _bfs_shortest_distances(edges, start)
        reachable_cands = [c for c in candidates if c in dists]
        assert reachable_cands == [target], f"Expected unique target {target}, got {reachable_cands}"
        assert dists[target] == d, f"Expected exact hop depth {d}, got {dists[target]}"

        edge_str = ", ".join(f"{u}->{v}" for u, v in edges)
        cand_str = ", ".join(candidates)
        prompt = f"DAG: Edges: {edge_str}; Start: {start}; Cand: {cand_str}; Reach:<ans>"
        full_text = prompt + target
        samples.append(
            Sample(
                prompt=prompt,
                target=target,
                full_text=full_text,
                regime="algorithmic",
                task="dag_planning",
                depth=d,
                answer_span_start=len(prompt),
            )
        )
        local_seen.add(sig)

    if len(samples) < n:
        raise RuntimeError(
            f"Could only generate {len(samples)}/{n} unique DAG samples after {attempts} attempts"
        )
    return samples


def _execute_python_trace(code_lines: Sequence[str], query_var: str) -> int:
    """Execute a restricted deterministic register trace in an isolated namespace."""
    env: dict[str, Any] = {}
    program = "\n".join(code_lines)
    exec(compile(program, "<python_trace>", "exec"), {"__builtins__": {}}, env)
    return int(env[query_var])


def generate_python_trace(
    n: int,
    depths: Sequence[int] = (2, 4),
    seed: int = 42,
    exclude_signatures: set[str] | None = None,
) -> list[Sample]:
    """Generate deterministic stateful Python register programs with `depth` transitions.

    Every program initializes registers `a, b, c`, executes exactly `depth`
    stateful arithmetic / conditional transitions modulo 17, and queries one
    register via `print(reg)`. Ground truth is verified by exact Python execution.
    """
    if n <= 0:
        return []
    if not depths or any(d < 1 for d in depths):
        raise ValueError(f"depths must be non-empty positive ints, got {depths}")

    rng = random.Random(seed)
    excluded = exclude_signatures if exclude_signatures is not None else set()
    local_seen: set[str] = set()
    samples: list[Sample] = []
    mod = 17
    regs = ("a", "b", "c")

    attempts = 0
    max_attempts = max(n * 25, 500)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        d = int(depths[len(samples) % len(depths)])
        a0 = rng.randint(0, mod - 1)
        b0 = rng.randint(0, mod - 1)
        c0 = rng.randint(0, mod - 1)

        code_lines: list[str] = [f"a={a0}", f"b={b0}", f"c={c0}"]
        last_written = "a"
        for step in range(d):
            op_kind = rng.randint(0, 4)
            dst = regs[step % 3] if step < d - 1 else rng.choice(regs)
            src1, src2 = rng.sample(regs, 2)
            k = rng.randint(1, 7)

            if op_kind == 0:
                stmt = f"{dst}=({src1}+{src2})%{mod}"
            elif op_kind == 1:
                stmt = f"{dst}=({src1}*2+{k})%{mod}"
            elif op_kind == 2:
                stmt = f"{dst}=({src1}-{src2}+{k})%{mod}"
            elif op_kind == 3:
                stmt = f"if {src1}<{src2}: {dst}=({dst}+{src1}+{k})%{mod}"
            else:
                stmt = f"if ({src1}+{src2})%2==0: {dst}=({dst}+{k})%{mod}"
            code_lines.append(stmt)
            last_written = dst

        query_var = last_written
        sig = f"py|{d}|{';'.join(code_lines)}|{query_var}"
        if sig in excluded or sig in local_seen:
            continue

        result = _execute_python_trace(code_lines, query_var)
        target = str(result)
        body = "; ".join(code_lines)
        prompt = f"Trace: {body}; print({query_var})<ans>"
        full_text = prompt + target

        samples.append(
            Sample(
                prompt=prompt,
                target=target,
                full_text=full_text,
                regime="algorithmic",
                task="python_trace",
                depth=d,
                answer_span_start=len(prompt),
            )
        )
        local_seen.add(sig)

    if len(samples) < n:
        raise RuntimeError(
            f"Could only generate {len(samples)}/{n} unique python_trace samples"
        )
    return samples


def generate_modular_varbind(
    n: int,
    depths: Sequence[int] = (2, 4),
    seed: int = 42,
    exclude_signatures: set[str] | None = None,
) -> list[Sample]:
    """Generate multi-step modular arithmetic & variable binding chains of exact depth `d`.

    Constructs a dependency chain `u_0 -> u_1 -> ... -> u_d` over randomly named
    variables `v0..v15` modulo 19, where `u_0` is a base integer and each `u_i`
    is an affine/modular function of `u_{i-1}` (with occasional earlier-variable
    dependencies that never shorten the chain depth). Binding equations are
    shuffled in the prompt so the model must resolve all `d` hops.
    """
    if n <= 0:
        return []
    if not depths or any(d < 1 for d in depths):
        raise ValueError(f"depths must be non-empty positive ints, got {depths}")

    rng = random.Random(seed)
    excluded = exclude_signatures if exclude_signatures is not None else set()
    local_seen: set[str] = set()
    samples: list[Sample] = []
    mod = 19
    var_pool = [f"v{i}" for i in range(16)]

    attempts = 0
    max_attempts = max(n * 25, 500)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        d = int(depths[len(samples) % len(depths)])
        needed = d + 1
        pool = var_pool if needed <= len(var_pool) else [f"v{i}" for i in range(needed + 2)]
        chain_vars = rng.sample(pool, needed)

        values: dict[str, int] = {}
        eqs: list[str] = []

        base_val = rng.randint(1, mod - 1)
        values[chain_vars[0]] = base_val
        eqs.append(f"{chain_vars[0]}={base_val}")

        for i in range(1, needed):
            prev_var = chain_vars[i - 1]
            cur_var = chain_vars[i]
            prev_val = values[prev_var]
            op_kind = rng.randint(0, 2)
            k = rng.randint(1, 9)
            if op_kind == 0:
                val = (prev_val + k) % mod
                eq = f"{cur_var}=({prev_var}+{k})%{mod}"
            elif op_kind == 1:
                mult = rng.choice((2, 3))
                val = (prev_val * mult + k) % mod
                eq = f"{cur_var}=({prev_var}*{mult}+{k})%{mod}"
            else:
                val = (prev_val - k) % mod
                eq = f"{cur_var}=({prev_var}-{k})%{mod}"
            values[cur_var] = val
            eqs.append(eq)

        query_var = chain_vars[-1]
        shuffled_eqs = list(eqs)
        rng.shuffle(shuffled_eqs)
        sig = f"mod|{d}|{query_var}|{';'.join(sorted(eqs))}"
        if sig in excluded or sig in local_seen:
            continue

        target = str(values[query_var])
        prompt = f"Mod: {'; '.join(shuffled_eqs)}; Query: {query_var}<ans>"
        full_text = prompt + target
        samples.append(
            Sample(
                prompt=prompt,
                target=target,
                full_text=full_text,
                regime="algorithmic",
                task="modular_varbind",
                depth=d,
                answer_span_start=len(prompt),
            )
        )
        local_seen.add(sig)

    if len(samples) < n:
        raise RuntimeError(
            f"Could only generate {len(samples)}/{n} unique modular_varbind samples"
        )
    return samples


# ---------------------------------------------------------------------------
# Regime 2: Structured Natural / Code Stream + H100 Web Mixture Loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntDomain:
    """Inclusive integer range sampled with ``rng.randint(lo, hi)``."""

    lo: int
    hi: int

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1

    def values(self) -> Sequence[int]:
        return range(self.lo, self.hi + 1)

    def draw(self, rng: random.Random) -> int:
        return rng.randint(self.lo, self.hi)


@dataclass(frozen=True)
class ChoiceDomain:
    """Finite set of string options sampled with ``rng.choice(options)``."""

    options: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.options)

    def values(self) -> Sequence[str]:
        return self.options

    def draw(self, rng: random.Random) -> str:
        return rng.choice(self.options)


@dataclass(frozen=True)
class TemplateParam:
    """One template parameter: the domain it was first sampled from, and a wider superset."""

    name: str
    original: IntDomain | ChoiceDomain
    widened: IntDomain | ChoiceDomain


@dataclass(frozen=True)
class NaturalTemplate:
    """A Regime 2 prompt/target template over an ordered tuple of parameters.

    Parameters are drawn in tuple order, so the draw sequence (and hence the
    generated data for a given seed) is fixed by this declaration. Every
    parameter combination must render to a distinct (prompt, target) pair.
    """

    name: str
    params: tuple[TemplateParam, ...]
    render: Callable[..., tuple[str, str]]

    def capacity(self, widened: bool) -> int:
        """Number of distinct samples the template can produce."""
        size = 1
        for p in self.params:
            size *= (p.widened if widened else p.original).size
        return size

    def draw(self, rng: random.Random, widened: bool) -> tuple[str, str]:
        values = {p.name: (p.widened if widened else p.original).draw(rng) for p in self.params}
        return self.render(**values)

    def original_signatures(self) -> set[str]:
        """Signatures of every sample in the original domain (enumerated, no sampling)."""
        names = [p.name for p in self.params]
        return {
            _natural_signature(*self.render(**dict(zip(names, combo))))
            for combo in itertools.product(*(p.original.values() for p in self.params))
        }


def _natural_signature(prompt: str, target: str) -> str:
    return f"nat|{prompt}|{target}"


def _render_math_groups(a: int, b: int, c: int, item: str) -> tuple[str, str]:
    prod = a * b
    prompt = (
        f"Math: Given {a} groups of {b} {item} plus {c} extra {item}. "
        f"Compute total = {a} * {b} + {c}. Then <ans>"
    )
    return prompt, f"{prod} + {c} = {prod + c}"


def _render_code_eval(fn: str, m: int, k: int, x0: int, y0: int) -> tuple[str, str]:
    prompt = (
        f"Code: def {fn}(x, y): return {m}*x + y + {k}\n"
        f"Note: Evaluate {fn}({x0}, {y0}). Output:<ans>"
    )
    return prompt, f"{m}*{x0} + {y0} + {k} = {m * x0 + y0 + k}"


def _render_arithmetic_series(start: int, step: int, n_terms: int) -> tuple[str, str]:
    seq = [start + i * step for i in range(n_terms + 2)]
    shown = ", ".join(str(v) for v in seq[:n_terms])
    prompt = f"Math: Arithmetic series with step {step}: {shown}. Next two terms:<ans>"
    return prompt, f"{seq[n_terms]}, {seq[n_terms + 1]}"


_NAT_ITEMS = ChoiceDomain(("crates", "packets", "blocks", "tokens", "nodes", "frames", "tiles", "cells"))
_NAT_FN_NAMES = ChoiceDomain((
    "scale_sum", "linear_step", "box_metric", "shift_total",
    "weighted_val", "poly_eval", "pair_score", "delta_norm",
))

# Round-robin order matters: sample i comes from template i % len(NATURAL_TEMPLATES).
# The `original` domains are the ones every committed dataset (EXP-001, the published
# HF splits) was generated from; changing them, or the parameter order, changes that data.
NATURAL_TEMPLATES: tuple[NaturalTemplate, ...] = (
    NaturalTemplate(
        name="math_groups",
        params=(
            TemplateParam("a", IntDomain(2, 19), IntDomain(2, 49)),
            TemplateParam("b", IntDomain(2, 15), IntDomain(2, 49)),
            TemplateParam("c", IntDomain(1, 12), IntDomain(1, 49)),
            TemplateParam("item", _NAT_ITEMS, _NAT_ITEMS),
        ),
        render=_render_math_groups,
    ),
    NaturalTemplate(
        name="code_eval",
        params=(
            TemplateParam("fn", _NAT_FN_NAMES, _NAT_FN_NAMES),
            TemplateParam("m", IntDomain(2, 9), IntDomain(2, 19)),
            TemplateParam("k", IntDomain(1, 15), IntDomain(1, 49)),
            TemplateParam("x0", IntDomain(1, 12), IntDomain(1, 49)),
            TemplateParam("y0", IntDomain(1, 12), IntDomain(1, 49)),
        ),
        render=_render_code_eval,
    ),
    NaturalTemplate(
        name="arithmetic_series",
        params=(
            TemplateParam("start", IntDomain(1, 15), IntDomain(1, 499)),
            TemplateParam("step", IntDomain(2, 9), IntDomain(2, 49)),
            TemplateParam("n_terms", IntDomain(3, 5), IntDomain(3, 5)),
        ),
        render=_render_arithmetic_series,
    ),
)


def generate_natural_stream(
    n: int,
    seed: int = 42,
    exclude_signatures: set[str] | None = None,
) -> list[Sample]:
    """Deterministic structured synthetic math-prose & Python code stream for CPU runs.

    Provides multi-token structured continuations (4–12 tokens) suitable for
    non-myopic `n_ahead=4` RL rewards and natural-text calibration evaluation.

    Sample ``i`` comes from ``NATURAL_TEMPLATES[i % len(NATURAL_TEMPLATES)]``, drawn
    from the template's original parameter domains with duplicate rejection. A
    template switches to its widened domains only once every sample in its original
    domain is used or excluded. Until then the draws are exactly those of the
    original generator, so any request it could fill returns identical samples.
    """
    if n <= 0:
        return []

    k = len(NATURAL_TEMPLATES)
    for t_idx, template in enumerate(NATURAL_TEMPLATES):
        quota = max(0, (n - t_idx + k - 1) // k)  # indices t_idx, t_idx + k, ... below n
        cap = template.capacity(widened=True)
        if quota > cap:
            raise ValueError(
                f"generate_natural_stream(n={n}) needs {quota} samples from template "
                f"{template.name!r}, which can produce at most {cap}"
            )

    rng = random.Random(seed)
    excluded = exclude_signatures if exclude_signatures is not None else set()
    local_seen: set[str] = set()
    samples: list[Sample] = []
    # Unused original-domain samples per template; None until first computed. It is
    # computed (by enumeration, without touching `rng`) at the template's first duplicate.
    unused_original: list[int | None] = [None] * k

    attempts = 0
    max_attempts = max(n * 25, 500)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        t_idx = len(samples) % k
        template = NATURAL_TEMPLATES[t_idx]
        widened = unused_original[t_idx] == 0
        prompt, target = template.draw(rng, widened=widened)
        sig = _natural_signature(prompt, target)
        if sig in excluded or sig in local_seen:
            if unused_original[t_idx] is None:
                unused_original[t_idx] = sum(
                    1
                    for s in template.original_signatures()
                    if s not in excluded and s not in local_seen
                )
            continue
        if not widened and unused_original[t_idx] is not None:
            unused_original[t_idx] -= 1

        samples.append(
            Sample(
                prompt=prompt,
                target=target,
                full_text=prompt + target,
                regime="natural",
                task="natural_code_math",
                depth=0,
                answer_span_start=len(prompt),
            )
        )
        local_seen.add(sig)

    if len(samples) < n:
        raise RuntimeError(
            f"Could only generate {len(samples)}/{n} unique natural_stream samples "
            f"in {max_attempts} attempts"
        )
    return samples


def load_h100_web_mixture(
    seed: int = 42,
    split: str = "train",
    streaming: bool = True,
    weights: tuple[float, float, float] = (0.40, 0.40, 0.20),
    dataset_loader: Callable[..., Any] | None = None,
) -> Any:
    """Build the Track 1 §5.2 H100 Regime 2 streaming corpus mixture:

      - 40% OpenWebMath (`open-web-math/open-web-math`)
      - 40% The Stack v2 Python/C++ (`bigcode/the-stack-v2-dedup`)
      - 20% FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)

    Requires the optional `[h100]` dependencies (`datasets`). Passing a custom
    `dataset_loader` allows offline unit testing without network calls.
    """
    if len(weights) != 3 or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must be a 3-tuple summing to 1.0, got {weights}")

    if dataset_loader is None:
        try:
            from datasets import interleave_datasets, load_dataset
        except ImportError as exc:
            raise ImportError(
                "load_h100_web_mixture requires optional [h100] dependencies "
                "(install via `uv sync --extra h100`)."
            ) from exc

        owm = load_dataset(
            "open-web-math/open-web-math",
            split=split,
            streaming=streaming,
        )
        stack = load_dataset(
            "bigcode/the-stack-v2-dedup",
            data_dir="data/Python",
            split=split,
            streaming=streaming,
        )
        fineweb = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=split,
            streaming=streaming,
        )
        return interleave_datasets(
            [owm, stack, fineweb],
            probabilities=list(weights),
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    specs = (
        ("open-web-math/open-web-math", None, weights[0]),
        ("bigcode/the-stack-v2-dedup", "data/Python", weights[1]),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", weights[2]),
    )
    streams = [
        dataset_loader(name, subset=subset, split=split, streaming=streaming)
        for name, subset, _ in specs
    ]
    return {
        "streams": streams,
        "weights": list(weights),
        "seed": seed,
        "sources": [s[0] for s in specs],
    }


# ---------------------------------------------------------------------------
# Regime 3: Aleatoric Negative Control (Irreducible Entropy)
# ---------------------------------------------------------------------------


_ENTITY_NAMES: tuple[str, ...] = (
    "Zara", "Kael", "Mira", "Orin", "Lyra", "Taro", "Nadia", "Bram",
    "Soren", "Veda", "Ciro", "Elena", "Dax", "Hana", "Ivo", "Juno",
    "Kira", "Luka", "Milo", "Nora", "Otto", "Pia", "Quin", "Rhea",
)
_ATTR_KINDS: tuple[str, ...] = ("badge", "vault", "label", "group")


def generate_aleatoric_cloze(
    n: int,
    seed: int = 42,
    num_classes: int = 16,
    namespace: str = "eval",
) -> list[Sample]:
    """Generate irreducible-entropy entity-attribute cloze negative controls.

    Each sample assigns an i.i.d. uniform class `C00..C{num_classes-1}` to a
    unique entity identifier (`<Name>_<namespace>_<unique_id>`). Because no rule
    connects the prompt to the target, true conditional entropy is strictly
    `log2(num_classes)` bits (4.0 bits for default `num_classes=16`).
    """
    if n <= 0:
        return []
    if num_classes < 2 or num_classes > 64:
        raise ValueError(f"num_classes must be in [2, 64], got {num_classes}")

    rng = random.Random(seed)
    classes = [f"C{i:02d}" for i in range(num_classes)]
    samples: list[Sample] = []

    # Generate unique shuffled entity IDs so no entity ever repeats within a split.
    id_offset = int(hashlib.sha256(f"{namespace}:{seed}".encode()).hexdigest()[:6], 16) % 500_000
    entity_nums = list(range(id_offset + 1000, id_offset + 1000 + n))
    rng.shuffle(entity_nums)

    for i in range(n):
        name = rng.choice(_ENTITY_NAMES)
        attr = rng.choice(_ATTR_KINDS)
        entity = f"{name}_{namespace}_{entity_nums[i]}"
        target = rng.choice(classes)
        prompt = f"Entity: {entity} | Attr: {attr} | Query: {attr}({entity})<ans>"
        samples.append(
            Sample(
                prompt=prompt,
                target=target,
                full_text=prompt + target,
                regime="aleatoric",
                task="aleatoric_cloze",
                depth=0,
                answer_span_start=len(prompt),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Tokenization, Dataset Wrapper, Collation, and Stratified Split Builder
# ---------------------------------------------------------------------------


def tokenize_sample(
    sample: Sample,
    tokenizer: PilotTokenizer,
    max_seq_len: int = 256,
) -> TokenizedSample:
    """Tokenize a `Sample` into a `TokenizedSample` preserving exact prompt/target boundary."""
    prompt_ids = tokenizer.encode(sample.prompt, add_bos=False, add_eos=False)
    target_ids = tokenizer.encode(sample.target, add_bos=False, add_eos=False)

    # Ensure total length fits in max_seq_len (reserving <bos>, target_ids, <eos>).
    max_prompt_tokens = max(1, max_seq_len - len(target_ids) - 2)
    if len(prompt_ids) > max_prompt_tokens:
        # Keep the tail of the prompt (which ends with `<ans>`) so the answer trigger survives.
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    input_ids = [tokenizer.bos_id] + prompt_ids + target_ids + [tokenizer.eos_id]
    prompt_len = 1 + len(prompt_ids)
    return TokenizedSample(
        input_ids=input_ids,
        prompt_len=prompt_len,
        target_ids=target_ids,
        regime=sample.regime,
        task=sample.task,
        depth=sample.depth,
    )


class PilotDataset(Dataset):
    """PyTorch Dataset holding both raw `Sample` and tokenized `TokenizedSample` records."""

    def __init__(
        self,
        samples: Sequence[Sample],
        tokenizer: PilotTokenizer,
        max_seq_len: int = 256,
    ) -> None:
        self.samples: list[Sample] = list(samples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenized: list[TokenizedSample] = [
            tokenize_sample(s, tokenizer, max_seq_len=max_seq_len) for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> TokenizedSample:
        return self.tokenized[idx]


def collate_batch(
    batch: Sequence[TokenizedSample | dict[str, Any]],
    pad_id: int = 0,
) -> dict[str, Any]:
    """Collate a list of `TokenizedSample` (or dicts) into padded PyTorch tensors.

    Returns:
      - `input_ids`: `[B, T]` LongTensor padded with `pad_id`
      - `attention_mask`: `[B, T]` LongTensor (1 for valid tokens, 0 for pad)
      - `labels`: `[B, T]` LongTensor (`input_ids` with pad positions set to -100)
      - `target_mask`: `[B, T]` BoolTensor (`True` strictly on answer target positions)
      - `prompt_lens`: `[B]` LongTensor
      - `target_lens`: `[B]` LongTensor
      - `depths`: `[B]` LongTensor
      - `regimes`: `list[str]`
      - `tasks`: `list[str]`
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    items: list[TokenizedSample] = []
    for elem in batch:
        if isinstance(elem, TokenizedSample):
            items.append(elem)
        else:
            items.append(
                TokenizedSample(
                    input_ids=list(elem["input_ids"]),
                    prompt_len=int(elem["prompt_len"]),
                    target_ids=list(elem["target_ids"]),
                    regime=str(elem["regime"]),
                    task=str(elem["task"]),
                    depth=int(elem["depth"]),
                )
            )

    batch_size = len(items)
    max_len = max(len(it.input_ids) for it in items)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    target_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    prompt_lens = torch.zeros(batch_size, dtype=torch.long)
    target_lens = torch.zeros(batch_size, dtype=torch.long)
    depths = torch.zeros(batch_size, dtype=torch.long)

    for i, it in enumerate(items):
        seq_len = len(it.input_ids)
        seq_tensor = torch.tensor(it.input_ids, dtype=torch.long)
        input_ids[i, :seq_len] = seq_tensor
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = seq_tensor

        t_start = it.prompt_len
        t_end = min(seq_len, t_start + len(it.target_ids))
        if t_end > t_start:
            target_mask[i, t_start:t_end] = True

        prompt_lens[i] = it.prompt_len
        target_lens[i] = len(it.target_ids)
        depths[i] = it.depth

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_mask": target_mask,
        "prompt_lens": prompt_lens,
        "target_lens": target_lens,
        "depths": depths,
        "regimes": [it.regime for it in items],
        "tasks": [it.task for it in items],
    }


# Fractions of the `train` split drawn from Regimes 1 / 2 / 3 (algorithmic, natural, aleatoric).
TRAIN_REGIME_MIX: tuple[float, float, float] = (0.75, 0.15, 0.10)


def _split_counts(total: int, ratios: Sequence[float]) -> list[int]:
    """Deterministically partition `total` into integer counts matching `ratios`."""
    raw = [total * r for r in ratios]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _prompt_signature(sample: Sample) -> str:
    return f"{sample.task}|{sample.depth}|{sample.prompt}"


def _build_algorithmic_mix(
    n: int,
    depths: Sequence[int],
    seed: int,
    exclude_prompts: set[str] | None = None,
) -> list[Sample]:
    """Generate an even mix of DAG planning, Python trace, and modular varbind."""
    c_dag, c_py, c_mod = _split_counts(n, (1 / 3, 1 / 3, 1 / 3))
    # Build exclusion sets using generator-internal + prompt-level dedup.
    out: list[Sample] = []
    for gen_fn, count, offset in (
        (generate_dag_planning, c_dag, 101),
        (generate_python_trace, c_py, 202),
        (generate_modular_varbind, c_mod, 303),
    ):
        if count <= 0:
            continue
        # Oversample slightly if needed when excluding train prompts from eval_id.
        collected: list[Sample] = []
        attempt = 0
        while len(collected) < count:
            need = count - len(collected)
            batch = gen_fn(need + (16 if exclude_prompts else 0), depths=depths, seed=seed + offset + attempt * 997)
            for s in batch:
                sig = _prompt_signature(s)
                if exclude_prompts is not None and sig in exclude_prompts:
                    continue
                collected.append(s)
                if exclude_prompts is not None:
                    exclude_prompts.add(sig)
                if len(collected) == count:
                    break
            attempt += 1
            if attempt > 20:
                raise RuntimeError(f"Failed to sample {count} non-overlapping {gen_fn.__name__} examples")
        out.extend(collected)
    return out


def build_stratified_datasets(
    data_cfg: DataConfig,
    tokenizer: PilotTokenizer,
) -> dict[str, PilotDataset]:
    """Build all 5 stratified PyTorch datasets with guaranteed zero train/eval leakage.

    Splits:
      - `train`:          75% Regime 1 (ID depths) + 15% Regime 2 (natural) + 10% Regime 3 (aleatoric, disjoint entities)
      - `eval_id`:        Regime 1 at `data_cfg.train_depths` (strictly zero prompt overlap with `train`)
      - `eval_ood`:       Regime 1 at `data_cfg.eval_ood_depths` (strictly disjoint hop depths)
      - `eval_natural`:   Regime 2 structured math-prose/code (strictly zero overlap with `train`)
      - `eval_aleatoric`: Regime 3 irreducible-entropy cloze (strictly disjoint entity namespace)
    """
    overlap = set(data_cfg.train_depths) & set(data_cfg.eval_ood_depths)
    if overlap:
        raise ValueError(
            f"train_depths {data_cfg.train_depths} and eval_ood_depths {data_cfg.eval_ood_depths} "
            f"must be strictly disjoint; overlap={overlap}"
        )

    seed = data_cfg.seed
    n_train = data_cfg.n_train_samples
    n_eval = data_cfg.n_eval_samples

    n_train_alg, n_train_nat, n_train_alea = _split_counts(n_train, TRAIN_REGIME_MIX)

    seen_alg_prompts: set[str] = set()
    train_alg = _build_algorithmic_mix(
        n_train_alg,
        depths=data_cfg.train_depths,
        seed=seed + 1000,
        exclude_prompts=seen_alg_prompts,
    )
    train_nat = generate_natural_stream(n_train_nat, seed=seed + 2000)
    train_nat_sigs = {f"nat|{s.prompt}|{s.target}" for s in train_nat}
    train_alea = generate_aleatoric_cloze(
        n_train_alea,
        seed=seed + 3000,
        namespace="tr",
    )

    train_samples = train_alg + train_nat + train_alea
    rng = random.Random(seed + 4000)
    rng.shuffle(train_samples)

    # Eval splits (strictly disjoint from train).
    eval_id_samples = _build_algorithmic_mix(
        n_eval,
        depths=data_cfg.train_depths,
        seed=seed + 10_000,
        exclude_prompts=seen_alg_prompts,
    )
    eval_ood_samples = _build_algorithmic_mix(
        n_eval,
        depths=data_cfg.eval_ood_depths,
        seed=seed + 20_000,
        exclude_prompts=seen_alg_prompts,
    )
    eval_natural_samples = generate_natural_stream(
        n_eval,
        seed=seed + 30_000,
        exclude_signatures=train_nat_sigs,
    )
    eval_aleatoric_samples = generate_aleatoric_cloze(
        n_eval,
        seed=seed + 40_000,
        namespace="ev",
    )

    return {
        "train": PilotDataset(train_samples, tokenizer, max_seq_len=data_cfg.max_seq_len),
        "eval_id": PilotDataset(eval_id_samples, tokenizer, max_seq_len=data_cfg.max_seq_len),
        "eval_ood": PilotDataset(eval_ood_samples, tokenizer, max_seq_len=data_cfg.max_seq_len),
        "eval_natural": PilotDataset(eval_natural_samples, tokenizer, max_seq_len=data_cfg.max_seq_len),
        "eval_aleatoric": PilotDataset(eval_aleatoric_samples, tokenizer, max_seq_len=data_cfg.max_seq_len),
    }
