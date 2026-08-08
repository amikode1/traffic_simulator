"""Hybrid Scenario Generation and Search Algorithm — Phase 2 of the Braess solver.

Solves the NP-hard problem of finding the optimal subset of Braess-tainted
roads to close using a Genetic Algorithm (GA). Chromosomes are binary strings
where 1 = road closed, 0 = road open.

Two modules:
  Module 1 — Creating the Chromosomes Pool (random + merit-based)
  Module 2 — Breeding (parent selection, mating, offspring evaluation)

Usage:
    python -m src.genetic_algorithm "Greenwood_Township__Pennsylvania__United_States"
"""

import logging
import math
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

import config
from src.braess_detector import (
    BraessRegistry,
    BraessResult,
    HeadlessSimulation,
    _load_city,
    _generate_commuters,
)
from src.road_network import NodeID, EdgeKey

log = logging.getLogger(__name__)

# ── RNG (deterministic for reproducibility) ─────────────────────
_rng = random.Random(42)


# ── Chromosome ────────────────────────────────────────────────────

@dataclass
class Chromosome:
    """A candidate solution: a binary string of which roads to close.

    Attributes:
        genes: List of 0/1 values, one per Braess-tainted road.
        benefit: baseline_avg - test_avg (positive means improvement).
        num_closed: Number of roads closed (genes == 1).
    """
    genes: list[int]
    benefit: float = 0.0
    evaluated: bool = False

    @property
    def num_closed(self) -> int:
        """Count of closed roads in this chromosome."""
        return sum(self.genes)

    def __hash__(self) -> int:
        return hash(tuple(self.genes))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return self.genes == other.genes


# ── Parallel evaluation worker ────────────────────────────────────

def _evaluate_chromosome_worker(args: tuple) -> tuple[list[int], float]:
    """Evaluate a single chromosome by running headless simulation.

    Runs in a worker process. Loads the city independently to avoid
    pickling the RoadNetwork.

    Args:
        args: (genes, city_name, commuter_pairs, braess_edges,
               baseline_avg, min_completed, dt)

    Returns:
        (genes, benefit) where benefit = baseline_avg - test_avg.
    """
    (genes, city_name, commuter_pairs, braess_edges,
     baseline_avg, min_completed, dt) = args

    rn, _ = _load_city(city_name)

    # Block roads where gene == 1
    for i, gene in enumerate(genes):
        if gene == 1:
            u, v, key = braess_edges[i]
            rn.block_edge(u, v, key)

    # Run simulation
    sim = HeadlessSimulation(rn, poisson_rate=0.5)
    result = sim.run_until_stable(
        min_completed=min_completed,
        dt=dt,
        progress_callback=None,
    )
    test_avg = result["avg_travel_time_seconds"]
    benefit = baseline_avg - test_avg

    return (genes, benefit)


# ── Genetic Algorithm ─────────────────────────────────────────────

class GeneticAlgorithm:
    """Hybrid Scenario Generation and Search Algorithm.

    Finds the optimal subset of Braess-tainted roads to close using:
      Module 1 — Creating the Chromosomes Pool (Process 1: random,
                 Process 2: merit-based)
      Module 2 — Breeding (parent selection, mating, evaluation)

    Attributes:
        registry: BraessRegistry from Phase 1 (contains all road scores).
        road_network: The city road network (for reference only).
        commuter_pairs: Fixed origin-destination pairs for evaluation.
        n: Number of Braess-tainted roads (genes per chromosome).
        n_bar: Chromosome capacity (estimated closed roads in optimal solution).
        q: Initial pool size calculated from the 95% formula.
        mating_prob: Probability each gene is inherited from the strong parent.
        merit_power: Exponent for the volatility-reduction term.
        min_completed: Trips per simulation evaluation.
        dt: Simulation timestep.
        max_single_score: Best score from a single-road closure (Phase 1).
        pool: List of chromosomes (sorted by benefit descending).
        merit_points: dict[gene_index, total_collected_points].
        iteration: Current iteration counter (for volatility calculation).
        max_iterations: i_max = |S| (number of tainted roads).
        best_chromosome: The best chromosome found so far.
        iterations_since_improvement: Count of consecutive breed cycles
            without improvement (used for termination).
    """

    def __init__(
        self,
        registry: BraessRegistry,
        road_network: Any,  # RoadNetwork — not used directly by GA but needed for interface
        commuter_pairs: list[tuple[NodeID, NodeID]],
        capacity_ratio: float = 0.3,
        mating_prob: float = 0.7,
        merit_power: float = 2.0,
        min_completed: int = 500,
        dt: float = 0.2,
        parallel: bool = True,  # set False for tests
    ) -> None:
        self.registry = registry
        self.commuter_pairs = commuter_pairs
        self.parallel = parallel

        # Braess-tainted roads (the gene pool)
        self.braess_roads: list[BraessResult] = sorted(
            registry.braess_roads,
            key=lambda r: -r.score,  # highest score first
        )
        self.n = len(self.braess_roads)
        self.n_bar = max(1, int(self.n * capacity_ratio))
        self.q = self._calculate_pool_size()
        self.mating_prob = mating_prob
        self.merit_power = merit_power
        self.min_completed = min_completed
        self.dt = dt

        # Entrance qualification threshold
        self.max_single_score = max(
            (r.score for r in self.braess_roads), default=0.0
        )

        # Pool
        self.pool: list[Chromosome] = []

        # Merit tracking
        self.merit_points: defaultdict[int, float] = defaultdict(float)
        # merit_points[gene_index] = sum of benefits of successful chromosomes
        # that include this road

        # Iteration tracking
        self.iteration: int = 0
        self.max_iterations: int = self.n  # i_max = |S|

        # Termination tracking
        self.best_chromosome: Optional[Chromosome] = None
        self.iterations_since_improvement: int = 0

        # Multiprocessing config
        self.num_workers: int = max(1, multiprocessing.cpu_count() - 1)

        # Timing for progress reporting
        self._process_start_time: float = 0.0

        # City identifier (for worker processes)
        self.city_name: str = registry.city_name

    # ── Pool size calculation ─────────────────────────────────────

    def _calculate_pool_size(self) -> int:
        """Calculate q: 1 - ((n - n̄) / n)^q = 0.95.

        Ensures every tainted road has a 95% chance of appearing in
        the initial random pool.

        Returns:
            Minimum pool size q (≥ 10).
        """
        if self.n <= self.n_bar:
            return max(10, self.n)
        ratio = (self.n - self.n_bar) / self.n
        if ratio <= 0.0:
            return 10
        q = math.log(0.05) / math.log(ratio)
        return max(10, int(math.ceil(q)))

    # ── Chromosome generation helpers ─────────────────────────────

    def _random_chromosome(self) -> Chromosome:
        """Generate a random chromosome with approximately n̄ roads closed.

        Assigns random values to each gene, sorts, and picks ~n̄ with
        the smallest values to close.

        Returns:
            A new unevaluated Chromosome.
        """
        # Assign random key to each road, sort, pick n̄ smallest
        road_indices = list(range(self.n))
        random.shuffle(road_indices)
        closed_set = set(road_indices[:self.n_bar])

        genes = [1 if i in closed_set else 0 for i in range(self.n)]
        return Chromosome(genes=genes)

    def _merit_based_chromosome(self) -> Chromosome:
        """Generate a chromosome by selecting roads via merit sampling.

        Roads with higher merit (collected_points + noise) are more
        likely to be selected for closure. The volatility term
        (1 - i/i_max)^power ensures early exploration.

        Returns:
            A new unevaluated Chromosome.
        """
        volatility = (1.0 - self.iteration / max(self.max_iterations, 1)) ** self.merit_power

        # Compute merit for each road
        merits = []
        for i in range(self.n):
            collected = self.merit_points[i]
            noise = _rng.uniform(0, 0.1)
            merit = (collected + noise) * volatility
            merits.append(max(merit, 1e-10))  # avoid zeros

        # Convert to probability distribution
        total = sum(merits)
        probs = [m / total for m in merits]

        # Sample ~n̄ roads without replacement
        num_to_close = max(1, min(self.n_bar, self.n))
        closed_indices = set()
        # Use roulette-wheel selection without replacement
        available = list(range(self.n))
        for _ in range(num_to_close):
            if not available:
                break
            # Create a sub-distribution over available indices
            avail_probs = [probs[i] for i in available]
            sub_total = sum(avail_probs)
            if sub_total <= 0:
                break
            avail_norm = [p / sub_total for p in avail_probs]
            chosen = _rng.choices(available, weights=avail_norm, k=1)[0]
            closed_indices.add(chosen)
            available.remove(chosen)

        genes = [1 if i in closed_indices else 0 for i in range(self.n)]
        return Chromosome(genes=genes)

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation: swap one 'closed' gene (1) with one 'open' gene (0).

        Args:
            chromosome: The chromosome to mutate.

        Returns:
            A new mutated Chromosome (original unchanged).
        """
        genes = list(chromosome.genes)
        # Find indices of closed (1) and open (0)
        closed_indices = [i for i, g in enumerate(genes) if g == 1]
        open_indices = [i for i, g in enumerate(genes) if g == 0]

        if closed_indices and open_indices:
            ci = _rng.choice(closed_indices)
            oi = _rng.choice(open_indices)
            genes[ci] = 0
            genes[oi] = 1

        return Chromosome(genes=genes)

    # ── Pool management ──────────────────────────────────────────

    def _is_duplicate(self, chromosome: Chromosome) -> bool:
        """Check if this chromosome already exists in the pool."""
        return any(chromosome == existing for existing in self.pool)

    def _add_to_pool(self, chromosome: Chromosome) -> None:
        """Add a chromosome to the pool and update merit tracking.

        Only adds if benefit > max_single_score (entrance qualification).

        Args:
            chromosome: The evaluated chromosome to add.

        Returns:
            True if added, False if it didn't qualify.
        """
        if not chromosome.evaluated:
            return
        if chromosome.benefit <= self.max_single_score:
            return

        self.pool.append(chromosome)

        # Update merit points: each closed road gets a share of the benefit
        if chromosome.benefit > 0 and chromosome.num_closed > 0:
            per_road_points = chromosome.benefit / chromosome.num_closed
            for i, gene in enumerate(chromosome.genes):
                if gene == 1:
                    self.merit_points[i] += per_road_points

        # Update best chromosome
        if (self.best_chromosome is None
                or chromosome.benefit > self.best_chromosome.benefit):
            self.best_chromosome = chromosome
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

    # ── Chromosome evaluation ────────────────────────────────────

    def _evaluate_chromosome(self, chromosome: Chromosome) -> Chromosome:
        """Evaluate a single chromosome by running headless simulation.

        Uses multiprocessing if available.

        Args:
            chromosome: The chromosome to evaluate.

        Returns:
            Evaluated Chromosome with benefit set.
        """
        if chromosome.evaluated:
            return chromosome

        braess_edges = [(r.edge[0], r.edge[1], r.edge[2])
                        for r in self.braess_roads]
        baseline_avg = self.registry.baseline_avg

        args = (
            chromosome.genes,
            self.city_name,
            self.commuter_pairs,
            braess_edges,
            baseline_avg,
            self.min_completed,
            self.dt,
        )

        genes, benefit = _evaluate_chromosome_worker(args)

        evaluated = Chromosome(genes=genes, benefit=benefit, evaluated=True)
        return evaluated

    def _evaluate_chromosomes_batch(
        self,
        chromosomes: list[Chromosome],
        verbose: bool = True,
    ) -> list[Chromosome]:
        """Evaluate multiple chromosomes.

        When parallel=True (default), uses multiprocessing with progress
        reporting. When parallel=False (test mode), evaluates serially
        so subclasses like FakeGA can override evaluation.

        Args:
            chromosomes: List of unevaluated chromosomes.
            verbose: Log progress.

        Returns:
            List of evaluated chromosomes.
        """
        if not chromosomes:
            return []

        n_jobs = len(chromosomes)

        # ── Serial path (test mode) ──
        if not self.parallel:
            evaluated: list[Chromosome] = []
            for c in chromosomes:
                evaluated.append(self._evaluate_chromosome(c))
            if verbose and evaluated:
                benefits = [c.benefit for c in evaluated]
                print(
                    f"  Evaluated {len(evaluated)} chromosomes: "
                    f"best benefit={max(benefits):.3f}s",
                    file=sys.stderr,
                )
            return evaluated

        # ── Parallel path (production) ──
        braess_edges = [(r.edge[0], r.edge[1], r.edge[2])
                        for r in self.braess_roads]
        baseline_avg = self.registry.baseline_avg

        worker_args = [
            (c.genes, self.city_name, self.commuter_pairs, braess_edges,
             baseline_avg, self.min_completed, self.dt)
            for c in chromosomes
        ]

        evaluated: list[Chromosome] = []
        progress_interval = max(1, n_jobs // 10)
        start_time = time.time()

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.imap_unordered(_evaluate_chromosome_worker, worker_args)
            for i, (genes, benefit) in enumerate(results, 1):
                evaluated.append(Chromosome(
                    genes=genes, benefit=benefit, evaluated=True,
                ))
                if verbose and (i % progress_interval == 0 or i == n_jobs):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (n_jobs - i) / rate if rate > 0 else 0
                    print(
                        f"  [{i}/{n_jobs}] "
                        f"({rate:.2f} eval/s, "
                        f"ETA: {eta:.0f}s)",
                        file=sys.stderr,
                    )

        if verbose and evaluated:
            benefits = [c.benefit for c in evaluated]
            best_idx = max(range(len(benefits)), key=lambda j: benefits[j])
            best_chrom = evaluated[best_idx]
            print(
                f"  Batch complete: best benefit={max(benefits):.3f}s "
                f"({best_chrom.num_closed} roads closed)",
                file=sys.stderr,
            )

        return evaluated

    # ── Module 1, Process 1: Initial Random Chromosomes ──────────

    def generate_initial_pool(
        self,
        verbose: bool = True,
    ) -> None:
        """Module 1, Process 1: Generate and evaluate q random chromosomes.

        Creates q random chromosomes, evaluates them in parallel, and
        adds those that pass the entrance qualification (benefit >
        max_single_score) to the pool.

        Args:
            verbose: Log progress.
        """
        if verbose:
            print(
                f"Process 1: Generating {self.q} random chromosomes...",
                file=sys.stderr,
            )

        # Generate q random chromosomes
        random_chromosomes = [self._random_chromosome() for _ in range(self.q)]

        # Evaluate in parallel
        evaluated = self._evaluate_chromosomes_batch(
            random_chromosomes, verbose=verbose,
        )

        # Add qualifying chromosomes to pool
        added = 0
        for chrom in evaluated:
            if chrom.benefit > self.max_single_score:
                self._add_to_pool(chrom)
                added += 1

        if verbose:
            print(
                f"  {added}/{self.q} chromosomes passed entrance "
                f"qualification (benefit > {self.max_single_score:.3f}s)",
                file=sys.stderr,
            )
            if self.best_chromosome:
                print(
                    f"  Best so far: benefit={self.best_chromosome.benefit:.3f}s "
                    f"({self.best_chromosome.num_closed} roads closed)",
                    file=sys.stderr,
                )

    # ── Module 1, Process 2: Advanced Chromosomes via Merit ──────

    def generate_advanced_chromosomes(
        self,
        target_pool_size: Optional[int] = None,
        verbose: bool = True,
    ) -> int:
        """Module 1, Process 2: Generate chromosomes via merit selection.

        Uses the Merit Index M_{i,s} to bias selection toward roads
        that have performed well in previous chromosomes. Batches
        evaluations for parallel speedup.

        Args:
            target_pool_size: Desired pool size (default: 2 * q).
            verbose: Log progress.

        Returns:
            Number of chromosomes added in this call.
        """
        if target_pool_size is None:
            target_pool_size = self.q * 2

        added = 0
        batch_size = max(1, self.num_workers * 2)  # keep workers fed
        safety_limit = target_pool_size * 5  # prevent infinite loop

        if verbose:
            print(
                f"Process 2: Generating merit-based chromosomes "
                f"(target pool size: {target_pool_size})...",
                file=sys.stderr,
            )

        remaining_needed = target_pool_size - len(self.pool)
        batches_done = 0

        while remaining_needed > 0 and self.iteration < self.max_iterations:
            # Generate a batch of merit-based chromosomes
            batch: list[Chromosome] = []
            safety = 0
            while len(batch) < batch_size and safety < batch_size * 3:
                safety += 1
                if self.iteration >= self.max_iterations:
                    break

                self.iteration += 1
                chrom = self._merit_based_chromosome()

                # Check for duplicates → mutate
                if self._is_duplicate(chrom):
                    chrom = self._mutate(chrom)
                    safeguard = 0
                    while self._is_duplicate(chrom) and safeguard < 10:
                        chrom = self._mutate(chrom)
                        safeguard += 1
                    if safeguard >= 10:
                        continue

                batch.append(chrom)

            if not batch:
                break

            # Evaluate entire batch in parallel
            start_batch = time.time()
            evaluated = self._evaluate_chromosomes_batch(
                batch, verbose=False,
            )

            # Add qualifying chromosomes to pool
            for chrom in evaluated:
                if chrom.benefit > self.max_single_score:
                    self._add_to_pool(chrom)
                    added += 1

            batches_done += 1
            remaining_needed = target_pool_size - len(self.pool)

            if verbose:
                elapsed = time.time() - start_batch
                print(
                    f"  Batch {batches_done}: {len(evaluated)} evals in "
                    f"{elapsed:.1f}s, added {sum(1 for c in evaluated if c.benefit > self.max_single_score)}, "
                    f"pool={len(self.pool)}/{target_pool_size}",
                    file=sys.stderr,
                )
                if self.best_chromosome:
                    print(
                        f"    Best: benefit={self.best_chromosome.benefit:.3f}s "
                        f"({self.best_chromosome.num_closed} roads closed)",
                        file=sys.stderr,
                    )

        if verbose:
            print(
                f"  Process 2 complete: added {added} merit-based chromosomes "
                f"(pool size: {len(self.pool)})",
                file=sys.stderr,
            )

        return added

    # ── Module 2: Breeding ───────────────────────────────────────

    def _select_parents(self) -> tuple[Chromosome, Chromosome]:
        """Select two parents for breeding.

        C1 (Ordinary Parent): Selected randomly from the entire pool.
        C2 (Strong Parent): Selected from the top 25% of the pool
            relative to C1 (i.e., positions closer to the best).

        Returns:
            (C1, C2) parents.
        """
        # Sort pool by benefit descending
        sorted_pool = sorted(self.pool, key=lambda c: -c.benefit)

        # Select C1 uniformly from the whole pool
        c1_index = _rng.randint(0, len(sorted_pool) - 1)
        c1 = sorted_pool[c1_index]

        # Select C2 from the top 25% of the range [0, c1_index]
        # This biases toward stronger chromosomes relative to C1
        top_range = max(1, c1_index // 4)
        c2_index = _rng.randint(0, min(top_range, len(sorted_pool) - 1))
        c2 = sorted_pool[c2_index]

        return c1, c2

    def _mate(self, c1: Chromosome, c2: Chromosome) -> Chromosome:
        """Create an offspring chromosome by mating two parents.

        For each gene, inherits from the strong parent (C2) with
        probability mating_prob, otherwise from the ordinary parent (C1).

        Args:
            c1: Ordinary parent.
            c2: Strong parent.

        Returns:
            A new unevaluated Chromosome.
        """
        genes = [
            c2.genes[i] if _rng.random() < self.mating_prob else c1.genes[i]
            for i in range(self.n)
        ]
        return Chromosome(genes=genes)

    def breed(self, max_iterations: int = 1000, verbose: bool = True) -> Chromosome:
        """Module 2: Breed the pool until termination.

        Repeatedly selects parents, mates, evaluates offspring (in
        batches), and adds to pool. Stops when no improvement is found
        for K consecutive iterations, where K = number of closed roads
        in the best chromosome.

        Args:
            max_iterations: Safety limit.
            verbose: Log progress.

        Returns:
            The best chromosome found.
        """
        if verbose:
            pool_size = len(self.pool)
            msg = (
                f"Module 2: Breeding (pool: {pool_size} chromosomes, "
                f"best benefit: {self.best_chromosome.benefit:.3f}s)"
                if self.best_chromosome else "Module 2: Breeding..."
            )
            print(msg, file=sys.stderr)

        start_time = time.time()
        batch_size = max(1, self.num_workers * 2)
        iteration = 0

        while iteration < max_iterations:
            if len(self.pool) < 2:
                if verbose:
                    print("  Pool too small for breeding.", file=sys.stderr)
                break

            # Check termination
            if self.best_chromosome is not None:
                k = self.best_chromosome.num_closed
                if k > 0 and self.iterations_since_improvement >= k:
                    if verbose:
                        elapsed = time.time() - start_time
                        print(
                            f"\nTermination: No improvement for {k} consecutive "
                            f"breeding iterations ({elapsed:.1f}s elapsed).",
                            file=sys.stderr,
                        )
                    break

            # Generate a batch of offspring
            offspring_batch: list[Chromosome] = []
            safety = 0
            while len(offspring_batch) < batch_size and safety < batch_size * 3:
                safety += 1
                if iteration >= max_iterations:
                    break
                iteration += 1

                # Select parents
                if len(self.pool) < 2:
                    break
                c1, c2 = self._select_parents()

                # Mate
                offspring = self._mate(c1, c2)

                # Check for duplicates → mutate
                if self._is_duplicate(offspring):
                    offspring = self._mutate(offspring)
                    safeguard = 0
                    while self._is_duplicate(offspring) and safeguard < 10:
                        offspring = self._mutate(offspring)
                        safeguard += 1
                    if safeguard >= 10:
                        continue

                offspring_batch.append(offspring)

            if not offspring_batch:
                break

            # Evaluate entire batch in parallel
            evaluated = self._evaluate_chromosomes_batch(
                offspring_batch, verbose=False,
            )

            # Add qualifying chromosomes to pool
            for chrom in evaluated:
                if chrom.benefit > self.max_single_score:
                    self._add_to_pool(chrom)
                else:
                    # Still count as a stall even if not qualified
                    if self.best_chromosome is not None:
                        self.iterations_since_improvement += 1

            # Progress logging
            if verbose and (iteration % max(1, max_iterations // 20) == 0 or iteration == max_iterations):
                elapsed = time.time() - start_time
                rate = iteration / elapsed if elapsed > 0 else 0
                k_info = ""
                if self.best_chromosome:
                    k = self.best_chromosome.num_closed
                    k_info = f"stall={self.iterations_since_improvement}/{k}"
                print(
                    f"  [{iteration}/{max_iterations}] "
                    f"pool={len(self.pool)}, "
                    f"best={self.best_chromosome.benefit:.3f}s "
                    f"({self.best_chromosome.num_closed} closed), "
                    f"{k_info} "
                    f"({rate:.1f} it/s)",
                    file=sys.stderr,
                )

        if verbose:
            elapsed = time.time() - start_time
            print(
                f"\nBreeding complete: {elapsed:.1f}s, "
                f"pool size: {len(self.pool)}",
                file=sys.stderr,
            )
            if self.best_chromosome:
                print(
                    f"Best chromosome: benefit={self.best_chromosome.benefit:.3f}s, "
                    f"{self.best_chromosome.num_closed} roads closed",
                    file=sys.stderr,
                )

        return self.best_chromosome

    # ── Full run ─────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> Chromosome:
        """Run the full Genetic Algorithm (Modules 1 + 2).

        1. Generate initial random pool.
        2. Generate merit-based chromosomes.
        3. Breed until termination.

        Args:
            verbose: Log progress.

        Returns:
            The best chromosome found.
        """
        total_start = time.time()

        if verbose:
            print(
                f"\n{'=' * 60}",
                file=sys.stderr,
            )
            print(
                f"Genetic Algorithm — Phase 2",
                file=sys.stderr,
            )
            print(
                f"  Braess-tainted roads: {self.n}",
                file=sys.stderr,
            )
            print(
                f"  Chromosome capacity (n̄): {self.n_bar}",
                file=sys.stderr,
            )
            print(
                f"  Initial pool size (q): {self.q}",
                file=sys.stderr,
            )
            print(
                f"  Baseline avg: {self.registry.baseline_avg:.2f}s",
                file=sys.stderr,
            )
            print(
                f"  Max single score: {self.max_single_score:.3f}s",
                file=sys.stderr,
            )
            print(
                f"{'=' * 60}",
                file=sys.stderr,
            )

        # Module 1, Process 1: Random pool
        self.generate_initial_pool(verbose=verbose)

        # Module 1, Process 2: Merit-based pool expansion
        if len(self.pool) < self.q * 2:
            self.generate_advanced_chromosomes(
                target_pool_size=self.q * 2,
                verbose=verbose,
            )

        # Module 2: Breeding
        final = self.breed(verbose=verbose)

        if verbose:
            total_elapsed = time.time() - total_start
            print(
                f"\nTotal GA time: {total_elapsed:.1f}s",
                file=sys.stderr,
            )

        return final


# ── CLI entry point ───────────────────────────────────────────────

def main() -> None:
    """Run the Genetic Algorithm from the command line.

    Expects a previously saved Braess registry file from Phase 1.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.genetic_algorithm <city_name> [commuters] [min_completed]", file=sys.stderr)
        print("Example: python -m src.genetic_algorithm Greenwood_Township__Pennsylvania__United_States", file=sys.stderr)
        sys.exit(1)

    city_name = sys.argv[1]
    commuter_count = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    min_completed = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    # Load the Braess registry from Phase 1
    registry_path = BraessRegistry.default_filename(city_name)
    if not os.path.exists(registry_path):
        print(f"Braess registry not found: {registry_path}", file=sys.stderr)
        print(f"Run Phase 1 first: python -m src.braess_detector \"{city_name}\"", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Braess registry: {registry_path}", file=sys.stderr)
    registry = BraessRegistry.load(registry_path)

    if registry.num_braess_roads == 0:
        print("No Braess-tainted roads found in registry. Nothing to optimize.", file=sys.stderr)
        sys.exit(0)

    print(
        f"Registry: {registry.num_braess_roads} Braess-tainted roads "
        f"(baseline: {registry.baseline_avg:.2f}s)",
        file=sys.stderr,
    )

    # Load the road network and generate commuters
    print(f"Loading city: {city_name}", file=sys.stderr)
    road_network, traffic_lights = _load_city(city_name)
    print(
        f"Loaded: {road_network.node_count} nodes, "
        f"{road_network.edge_count} edges",
        file=sys.stderr,
    )

    print(f"Generating {commuter_count} commuter pairs...", file=sys.stderr)
    commuter_pairs = _generate_commuters(road_network, count=commuter_count)

    # Run the Genetic Algorithm
    ga = GeneticAlgorithm(
        registry=registry,
        road_network=road_network,
        commuter_pairs=commuter_pairs,
        min_completed=min_completed,
        dt=0.2,
    )

    best = ga.run(verbose=True)

    # ── Save results ──
    output_path = f"data/ga_result_{city_name}.json"
    import json

    # Convert best chromosome to a list of road edges
    braess_roads_list = [
        {
            "index": i,
            "edge": list(registry.braess_roads[i].edge),
            "score": registry.braess_roads[i].score,
            "close": bool(gene == 1),
        }
        for i, gene in enumerate(best.genes)
    ]

    closed_roads = [r for r in braess_roads_list if r["close"]]
    result_data = {
        "city_name": city_name,
        "baseline_avg": registry.baseline_avg,
        "benefit": best.benefit,
        "optimal_avg": registry.baseline_avg - best.benefit,
        "num_braess_roads": registry.num_braess_roads,
        "num_closed_roads": best.num_closed,
        "pool_size_used": len(ga.pool),
        "closed_roads": closed_roads,
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\nResults saved to: {output_path}", file=sys.stderr)
    print(f"Optimal solution: close {best.num_closed} roads, save {best.benefit:.3f}s", file=sys.stderr)
    print(f"Optimal average travel time: {registry.baseline_avg - best.benefit:.2f}s", file=sys.stderr)

    # Print closed roads summary
    print("\n=== OPTIMAL SOLUTION ===")
    for r in closed_roads:
        print(f"  Road {r['edge']}: Phase 1 score={r['score']:+.3f}s")
    print(f"\nTotal benefit: {best.benefit:.3f}s (from {best.num_closed} closed roads)")


if __name__ == "__main__":
    main()