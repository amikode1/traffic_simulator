"""Unit tests for the Genetic Algorithm (Phase 2) logic.

Tests GA mechanics in isolation (selection, mating, mutation, merit,
pool management, termination) without running real simulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import math
from dataclasses import dataclass, field
from typing import Optional

from src.genetic_algorithm import (
    Chromosome,
    GeneticAlgorithm,
)
from src.braess_detector import BraessRegistry, BraessResult


def _make_synthetic_registry(
    num_roads: int = 20,
    baseline_avg: float = 100.0,
    seed: int = 42,
) -> BraessRegistry:
    """Create a synthetic registry with mock Braess-tainted roads."""
    rng = random.Random(seed)
    results = []
    for i in range(num_roads):
        score = rng.uniform(1.0, 10.0)
        test_avg = baseline_avg - score
        results.append(BraessResult(
            edge=(i * 100, i * 100 + 1, 0),
            baseline_avg=baseline_avg,
            test_avg=test_avg,
            score=score,
            disconnected=False,
        ))

    return BraessRegistry(
        city_name="test_city",
        commuter_count=100,
        min_completed=100,
        baseline_avg=baseline_avg,
        total_roads_tested=100,
        results=results,
    )


class FakeGA(GeneticAlgorithm):
    """GA subclass that skips real simulation evaluation.

    Uses synthetic benefit calculation: each chromosome's benefit is
    estimated by summing the Phase-1 scores of its closed roads,
    divided by num_closed (simulating diminishing returns).
    """

    def __init__(self, *args, **kwargs):
        kwargs["parallel"] = False
        super().__init__(*args, **kwargs)
        self.evaluation_count = 0

    def _evaluate_chromosome(self, chromosome: Chromosome) -> Chromosome:
        """Fake evaluation: compute synthetic benefit from Phase 1 scores."""
        self.evaluation_count += 1

        # Sum the scores of all closed roads, with diminishing returns
        total = 0.0
        for i, gene in enumerate(chromosome.genes):
            if gene == 1:
                total += self.braess_roads[i].score

        # Diminishing returns: benefit grows logarithmically
        benefit = math.sqrt(total) * 2.0

        return Chromosome(genes=chromosome.genes, benefit=benefit, evaluated=True)


# ── Tests ─────────────────────────────────────────────────────────

def test_chromosome_properties():
    """Verify Chromosome dataclass and helper properties."""
    c = Chromosome(genes=[1, 0, 1, 0, 0, 1])
    assert c.num_closed == 3, f"Expected 3 closed, got {c.num_closed}"
    assert c.benefit == 0.0
    assert not c.evaluated

    c.evaluated = True
    c.benefit = 5.5
    assert c.evaluated
    assert c.benefit == 5.5
    print("  ✓ Chromosome properties")


def test_chromosome_hash_and_eq():
    """Verify Chromosome hashing and equality work for duplicate detection."""
    c1 = Chromosome(genes=[1, 0, 1])
    c2 = Chromosome(genes=[1, 0, 1])
    c3 = Chromosome(genes=[1, 1, 0])

    assert c1 == c2
    assert c1 != c3
    assert hash(c1) == hash(c2)
    assert hash(c1) != hash(c3)
    print("  ✓ Chromosome hash and equality")


def test_pool_size_calculation():
    """Verify the q formula produces reasonable pool sizes."""
    registry = _make_synthetic_registry(num_roads=20)
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
        capacity_ratio=0.3,
    )
    # n=20, n_bar=6, ratio=(20-6)/20=0.7
    # q = log(0.05) / log(0.7) ≈ 8.4 → 9
    expected_q = max(10, math.ceil(math.log(0.05) / math.log(14/20)))
    assert ga.q >= 10, f"Expected q >= 10, got {ga.q}"
    print(f"  ✓ Pool size: n={ga.n}, n_bar={ga.n_bar}, q={ga.q}")


def test_random_chromosome():
    """Verify random chromosomes have ~n̄ closed roads."""
    registry = _make_synthetic_registry(num_roads=50)
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
        capacity_ratio=0.3,
    )

    # Generate several random chromosomes and check stats
    closed_counts = []
    for _ in range(100):
        c = ga._random_chromosome()
        assert len(c.genes) == ga.n
        closed_counts.append(c.num_closed)

    avg_closed = sum(closed_counts) / len(closed_counts)
    # Should be close to n_bar
    assert abs(avg_closed - ga.n_bar) < 2.0, (
        f"Avg closed {avg_closed:.1f} far from n_bar {ga.n_bar}"
    )
    print(f"  ✓ Random chromosome: avg closed={avg_closed:.1f} (n_bar={ga.n_bar})")


def test_mutation():
    """Verify mutation swaps exactly one 1 and one 0."""
    registry = _make_synthetic_registry(num_roads=20)
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )

    # Create a chromosome with known pattern
    original = Chromosome(genes=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    mutated = ga._mutate(original)

    # Should have same number of closed roads
    assert mutated.num_closed == original.num_closed, (
        f"Mutation changed closed count: {original.num_closed} → {mutated.num_closed}"
    )

    # Should be different from original
    assert mutated.genes != original.genes, "Mutation didn't change genes"

    # Count differences: should be exactly 2 (one 1→0, one 0→1)
    diffs = sum(1 for a, b in zip(original.genes, mutated.genes) if a != b)
    assert diffs == 2, f"Expected 2 differences, got {diffs}"
    print("  ✓ Mutation (swaps 1 and 0 correctly)")


def test_duplicate_detection():
    """Verify pool does not accept duplicates."""
    registry = _make_synthetic_registry(num_roads=10)
    # Set all Phase-1 scores to low values so entrance threshold is low
    for r in registry.results:
        r.score = 1.0
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )
    # max_single_score is now 1.0, so chromosome with benefit > 1.0 qualifies

    c1 = Chromosome(genes=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], benefit=5.0, evaluated=True)
    c2 = Chromosome(genes=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], benefit=5.0, evaluated=True)

    ga._add_to_pool(c1)
    assert len(ga.pool) == 1, f"Expected 1, got {len(ga.pool)}"
    assert ga._is_duplicate(c2)
    print("  ✓ Duplicate detection")


def test_entrance_qualification():
    """Verify chromosomes below max_single_score are rejected."""
    registry = _make_synthetic_registry(num_roads=10)
    # Make all scores very high
    for r in registry.results:
        r.score = 100.0
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )
    assert ga.max_single_score == 100.0

    # A chromosome with benefit > 100 should be accepted
    good = Chromosome(genes=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], benefit=150.0, evaluated=True)
    ga._add_to_pool(good)
    assert len(ga.pool) == 1

    # A chromosome with benefit <= 100 should be rejected
    bad = Chromosome(genes=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], benefit=50.0, evaluated=True)
    ga._add_to_pool(bad)
    assert len(ga.pool) == 1, "Bad chromosome was added to pool!"
    print("  ✓ Entrance qualification")


def test_merit_based_selection():
    """Verify merit-based chromosomes favour high-merit roads."""
    registry = _make_synthetic_registry(num_roads=20)
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )

    # Assign high merit to road 0, low to others
    ga.merit_points[0] = 1000.0
    for i in range(1, 20):
        ga.merit_points[i] = 1.0

    # Generate several merit-based chromosomes
    road0_selected = 0
    for _ in range(100):
        c = ga._merit_based_chromosome()
        if c.genes[0] == 1:
            road0_selected += 1

    # Road 0 should be selected more often than chance (~70%)
    assert road0_selected > 50, f"Road 0 selected only {road0_selected}/100 times"
    print(f"  ✓ Merit selection: road 0 chosen {road0_selected}/100 times")


def test_parent_selection():
    """Verify parent selection picks stronger chromosome as C2."""
    registry = _make_synthetic_registry(num_roads=10)
    # Set all Phase-1 scores low so entrance threshold is low
    for r in registry.results:
        r.score = 1.0
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )

    # Add chromosomes with known benefits (all > max_single_score)
    for benefit, genes in [
        (1.5, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (2.0, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        (3.0, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        (4.0, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        (5.0, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ]:
        c = Chromosome(genes=genes, benefit=benefit, evaluated=True)
        ga._add_to_pool(c)

    assert len(ga.pool) >= 5, f"Pool only has {len(ga.pool)} chromosomes"

    # Run many parent selections
    c2_higher_count = 0
    for _ in range(500):
        c1, c2 = ga._select_parents()
        # C2 should typically have higher benefit than C1
        if c2.benefit >= c1.benefit:
            c2_higher_count += 1

    assert c2_higher_count > 300, (
        f"C2 was not consistently stronger: {c2_higher_count}/500"
    )
    print(f"  ✓ Parent selection: C2 stronger {c2_higher_count}/500 times")


def test_mating():
    """Verify offspring inherits majority of genes from strong parent."""
    registry = _make_synthetic_registry(num_roads=20)
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
        mating_prob=0.7,
    )

    # Create two very different parents
    p1 = Chromosome(genes=[1] * 10 + [0] * 10)  # first half closed
    p2 = Chromosome(genes=[0] * 10 + [1] * 10)  # second half closed

    # Mate many times and check inheritance bias
    p2_genes_count = 0
    total_genes = 0
    for _ in range(500):
        offspring = ga._mate(p1, p2)
        for i in range(20):
            total_genes += 1
            if offspring.genes[i] == p2.genes[i]:
                p2_genes_count += 1

    p2_inheritance_rate = p2_genes_count / total_genes
    # Should be close to mating_prob (0.7)
    assert 0.6 < p2_inheritance_rate < 0.8, (
        f"P2 inheritance rate {p2_inheritance_rate:.2f} not near 0.7"
    )
    print(f"  ✓ Mating: P2 inheritance rate {p2_inheritance_rate:.2f} (expected ~0.7)")


def test_termination():
    """Verify breeding terminates after K iterations without improvement."""
    registry = _make_synthetic_registry(num_roads=10)
    # Set all Phase-1 scores low so entrance threshold is low
    for r in registry.results:
        r.score = 1.0
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
    )

    # Add several chromosomes to the pool (need at least 2 for breeding)
    chromosomes = [
        Chromosome(genes=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], benefit=100.0, evaluated=True),
        Chromosome(genes=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], benefit=90.0, evaluated=True),
    ]
    for c in chromosomes:
        ga._add_to_pool(c)
    assert len(ga.pool) >= 2, f"Pool too small: {len(ga.pool)}"
    assert ga.best_chromosome is not None
    k = ga.best_chromosome.num_closed  # K = num closed in best = 1

    # Run breeding — should terminate after K=1 stall iteration
    # because no new chromosome can beat 100.0
    best = ga.breed(max_iterations=100, verbose=False)
    assert best is not None
    assert best.benefit == 100.0, f"Expected 100.0, got {best.benefit}"
    assert ga.iterations_since_improvement >= k, (
        f"Didn't reach termination: {ga.iterations_since_improvement} < {k}"
    )
    print(f"  ✓ Termination: stopped after {ga.iterations_since_improvement} stall iterations "
          f"(K={k})")


def test_full_run():
    """Run the full GA pipeline end-to-end with fake evaluations."""
    registry = _make_synthetic_registry(num_roads=15)
    # Set all Phase-1 scores low so entrance threshold is low
    for r in registry.results:
        r.score = 2.0
    ga = FakeGA(
        registry=registry,
        road_network=None,
        commuter_pairs=[(1, 2)],
        capacity_ratio=0.3,
        mating_prob=0.7,
        min_completed=5,
    )

    # Process 1: Random pool
    ga.generate_initial_pool(verbose=False)
    assert len(ga.pool) > 0, "Pool is empty after Process 1"
    assert ga.best_chromosome is not None, "No best chromosome after Process 1"

    # Process 2: Merit-based
    if len(ga.pool) < ga.q * 2:
        ga.generate_advanced_chromosomes(verbose=False)
    assert len(ga.pool) >= ga.q, (
        f"Pool too small after Process 2: {len(ga.pool)} < {ga.q}"
    )

    # Breeding
    best = ga.breed(max_iterations=50, verbose=False)
    assert best is not None
    assert best.benefit > 0, f"Best benefit is not positive: {best.benefit}"
    assert best.evaluated, "Best chromosome not evaluated"
    print(f"  ✓ Full pipeline: pool={len(ga.pool)}, "
          f"evaluations={ga.evaluation_count}, "
          f"best benefit={best.benefit:.2f}s")


if __name__ == "__main__":
    print("Testing Genetic Algorithm logic...")
    print()
    test_chromosome_properties()
    test_chromosome_hash_and_eq()
    test_pool_size_calculation()
    test_random_chromosome()
    test_mutation()
    test_duplicate_detection()
    test_entrance_qualification()
    test_merit_based_selection()
    test_parent_selection()
    test_mating()
    test_termination()
    test_full_run()
    print("\nAll tests passed! ✓")