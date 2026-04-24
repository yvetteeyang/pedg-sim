"""
Prisoner Escape Dilemma Game — MVP Simulation.

Implements the mechanism from Yang (2025, Section 4):
N portfolio workers with private expertise choose among four actions
(C/D/R/S) under incomplete information. Strategies and network structure
co-evolve. Trust is updated via Bayesian (Beta) inference.

Network semantics (paper, Section 4):
  - strong tie = graph edge; formed on successful C-C collaboration
  - weak tie  = entry in agent.saved; formed by referral (R) or defer (S)
  - topological distance is shortest path in the strong-tie graph

Metrics tracked (paper, Section 4 validation):
  - clustering coefficient CC
  - average shortest path length L
  - cooperation rate ∂
  - small-worldness σ = (CC/CC_rand) / (L/L_rand)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------- Configuration ----------
N_AGENTS = 1000
N_EXPERTISE = 10
N_ROUNDS = 300
METRIC_INTERVAL = 10
SEED = 42

# Prisoner's Dilemma payoffs (T > R > P > S)
R_REWARD, T_TEMPT, S_SUCK, P_PUN = 3.0, 5.0, 0.0, 1.0
COMPLEMENTARITY_BONUS = 2.0

# Decision thresholds on trust estimate (posterior Beta mean)
TRUST_C = 0.40   # cooperate above this (uninformed prior 0.5 → start cooperative)
TRUST_D = 0.30   # decline below this

# Encounter mixing probabilities
P_REFERRAL_QUERY = 0.30
P_REVISIT        = 0.40
# Remaining mass → random stranger

BETRAYAL_WEIGHT = 2  # one burn weighted as two successes in Beta updating


# ---------- Agent ----------
@dataclass
class Agent:
    id: int
    expertise: int
    coop_propensity: float                          # latent type, hidden
    beliefs: dict = field(default_factory=dict)     # partner_id -> (alpha, beta)
    saved: set = field(default_factory=set)         # weak ties
    payoff: float = 0.0

    def trust(self, other_id: int) -> float:
        a, b = self.beliefs.get(other_id, (1.0, 1.0))
        return a / (a + b)

    def update_belief(self, other_id: int, cooperated: bool) -> None:
        a, b = self.beliefs.get(other_id, (1.0, 1.0))
        if cooperated:
            a += 1.0
        else:
            b += BETRAYAL_WEIGHT
        self.beliefs[other_id] = (a, b)


# ---------- Decision rule (C / D / R / S) ----------
def decide_action(me: Agent, them: Agent, g: nx.Graph, agents: dict) -> str:
    t = me.trust(them.id)
    # Gossip: if no direct prior, blend with mutual strong-tie neighbors' priors
    if them.id not in me.beliefs:
        mutuals = set(g.neighbors(me.id)) & set(g.neighbors(them.id))
        vals = [agents[m].trust(them.id) for m in mutuals if them.id in agents[m].beliefs]
        if vals:
            t = 0.5 * t + 0.5 * float(np.mean(vals))

    complementary = me.expertise != them.expertise

    if t >= TRUST_C and complementary:
        return 'C'
    if t < TRUST_D:
        return 'D'
    # Middling trust: route to a neighbor (R) if I have one, else defer (S)
    if g.degree(me.id) > 0 and random.random() < 0.5:
        return 'R'
    return 'S'


# ---------- Target selection ----------
def pick_target(agent: Agent, g: nx.Graph, agents: dict) -> Agent | None:
    r = random.random()
    deg = g.degree(agent.id)

    # (1) Referral query — reach a 2nd-degree contact via a neighbor
    if deg > 0 and r < P_REFERRAL_QUERY:
        neighbor = random.choice(list(g.neighbors(agent.id)))
        second = [s for s in g.neighbors(neighbor)
                  if s != agent.id and not g.has_edge(agent.id, s)]
        if second:
            target_id = random.choice(second)
            agent.saved.add(target_id)
            agents[target_id].saved.add(agent.id)
            return agents[target_id]

    # (2) Revisit — known strong-tie neighbor or saved weak tie
    if r < P_REFERRAL_QUERY + P_REVISIT:
        pool = list(g.neighbors(agent.id)) + list(agent.saved)
        if pool:
            target_id = random.choice(pool)
            if target_id != agent.id:
                return agents[target_id]

    # (3) Random stranger
    target_id = random.randint(0, N_AGENTS - 1)
    return agents[target_id] if target_id != agent.id else None


# ---------- Resolution of one encounter ----------
def resolve(a: Agent, b: Agent, act_a: str, act_b: str,
            g: nx.Graph, agents: dict) -> str | None:

    def apply_referral(referrer: Agent, partner: Agent) -> None:
        candidates = [n for n in g.neighbors(referrer.id) if n != partner.id]
        if candidates:
            intro = random.choice(candidates)
            agents[intro].saved.add(partner.id)
            partner.saved.add(intro)

    if act_a == 'R':
        apply_referral(a, b)
    if act_b == 'R':
        apply_referral(b, a)

    if act_a == 'S':
        a.saved.add(b.id)
    if act_b == 'S':
        b.saved.add(a.id)

    # C/D payoff resolution — runs only if both sides engaged in the collab move
    if act_a in ('C', 'D') and act_b in ('C', 'D') and 'C' in (act_a, act_b):
        # Stated C is honored only with probability = intrinsic propensity
        real_a = 'C' if act_a == 'C' and random.random() < a.coop_propensity else 'D'
        real_b = 'C' if act_b == 'C' and random.random() < b.coop_propensity else 'D'

        bonus = COMPLEMENTARITY_BONUS if a.expertise != b.expertise else 0.0

        if real_a == 'C' and real_b == 'C':
            a.payoff += R_REWARD + bonus
            b.payoff += R_REWARD + bonus
            a.update_belief(b.id, True)
            b.update_belief(a.id, True)
            g.add_edge(a.id, b.id)
            a.saved.discard(b.id)
            b.saved.discard(a.id)
            return 'CC'
        if real_a == 'C' and real_b == 'D':
            a.payoff += S_SUCK
            b.payoff += T_TEMPT
            a.update_belief(b.id, False)
            return 'CD'
        if real_a == 'D' and real_b == 'C':
            a.payoff += T_TEMPT
            b.payoff += S_SUCK
            b.update_belief(a.id, False)
            return 'DC'
        a.payoff += P_PUN
        b.payoff += P_PUN
        return 'DD'
    return None


# ---------- Metrics ----------
def sampled_avg_path_length(g: nx.Graph, sample_sources: int = 50) -> float:
    comps = list(nx.connected_components(g))
    if not comps:
        return 0.0
    largest = max(comps, key=len)
    if len(largest) < 3:
        return 0.0
    sub = g.subgraph(largest)
    sources = random.sample(list(largest), min(sample_sources, len(largest)))
    dists = [d for u in sources
             for v, d in nx.single_source_shortest_path_length(sub, u).items()
             if u != v]
    return float(np.mean(dists)) if dists else 0.0


def small_worldness(cc: float, L: float, n: int, m: int) -> float:
    if m == 0 or L == 0:
        return 0.0
    k = 2 * m / n                    # mean degree
    if k <= 1:
        return 0.0
    cc_rand = k / (n - 1)
    L_rand = np.log(n) / np.log(k)
    if cc_rand == 0 or L_rand == 0:
        return 0.0
    return (cc / cc_rand) / (L / L_rand)


# ---------- Main simulation loop ----------
def run_simulation() -> tuple[nx.Graph, dict, dict]:
    random.seed(SEED)
    np.random.seed(SEED)

    agents = {
        i: Agent(id=i,
                 expertise=random.randint(0, N_EXPERTISE - 1),
                 coop_propensity=random.random())
        for i in range(N_AGENTS)
    }

    g = nx.Graph()
    g.add_nodes_from(range(N_AGENTS))

    history = {'round': [], 'cc': [], 'L': [], 'coop_rate': [],
               'edges': [], 'sigma': []}

    print(f"{'round':>5} | {'edges':>5} | {'CC':>5} | {'L':>5} | "
          f"{'coop':>4} | {'σ':>5} | {'C':>4} {'D':>4} {'R':>4} {'S':>4}")
    print('-' * 72)

    for r in range(N_ROUNDS):
        coop_attempts = coop_success = 0
        counts = {'C': 0, 'D': 0, 'R': 0, 'S': 0}

        order = list(agents.keys())
        random.shuffle(order)
        for i in order:
            agent = agents[i]
            target = pick_target(agent, g, agents)
            if target is None or target.id == i:
                continue
            act_a = decide_action(agent, target, g, agents)
            act_b = decide_action(target, agent, g, agents)
            counts[act_a] += 1
            counts[act_b] += 1
            outcome = resolve(agent, target, act_a, act_b, g, agents)
            if outcome is not None and 'C' in outcome:
                coop_attempts += 1
                if outcome == 'CC':
                    coop_success += 1

        if r % METRIC_INTERVAL == 0 or r == N_ROUNDS - 1:
            cc = nx.average_clustering(g)
            L = sampled_avg_path_length(g)
            coop_rate = coop_success / coop_attempts if coop_attempts else 0.0
            edges = g.number_of_edges()
            sigma = small_worldness(cc, L, N_AGENTS, edges)

            history['round'].append(r)
            history['cc'].append(cc)
            history['L'].append(L)
            history['coop_rate'].append(coop_rate)
            history['edges'].append(edges)
            history['sigma'].append(sigma)

            print(f"{r:>5d} | {edges:>5d} | {cc:>5.3f} | {L:>5.2f} | "
                  f"{coop_rate:>4.2f} | {sigma:>5.2f} | "
                  f"{counts['C']:>4d} {counts['D']:>4d} "
                  f"{counts['R']:>4d} {counts['S']:>4d}")

    return g, agents, history


# ---------- Plotting ----------
def plot_results(g: nx.Graph, agents: dict, history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(history['round'], history['cc'], color='tab:blue')
    ax.set_xlabel('round'); ax.set_ylabel('CC')
    ax.set_title('Clustering Coefficient')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history['round'], history['L'], color='tab:red')
    ax.set_xlabel('round'); ax.set_ylabel('L')
    ax.set_title('Average Shortest Path Length')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(history['round'], history['sigma'], color='tab:purple')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('round'); ax.set_ylabel('σ')
    ax.set_title('Small-Worldness σ\n(>1 = small-world regime)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history['round'], history['coop_rate'], color='tab:green')
    ax.set_ylim(0, 1)
    ax.set_xlabel('round'); ax.set_ylabel('coop rate ∂')
    ax.set_title('Cooperation Rate (C–C / C-attempts)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history['round'], history['edges'], color='tab:orange')
    ax.set_xlabel('round'); ax.set_ylabel('edges')
    ax.set_title('Strong Ties Formed')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    degrees = [d for _, d in g.degree()]
    ax.hist(degrees, bins=40, color='tab:gray', edgecolor='black')
    ax.set_xlabel('degree'); ax.set_ylabel('num agents')
    ax.set_title(f'Final Degree Distribution\n(edges={g.number_of_edges()})')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Prisoner Escape Dilemma Game — N={N_AGENTS} agents, {N_ROUNDS} rounds')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nplot saved → {out_path}")
    plt.show()


if __name__ == '__main__':
    g, agents, history = run_simulation()
    payoffs = [a.payoff for a in agents.values()]
    print('\n=== Final ===')
    print(f"edges:               {g.number_of_edges()}")
    print(f"components:          {nx.number_connected_components(g)}")
    print(f"isolated nodes:      {sum(1 for _, d in g.degree() if d == 0)}")
    print(f"mean payoff:         {np.mean(payoffs):.2f}")
    print(f"mean degree:         {np.mean([d for _, d in g.degree()]):.2f}")
    print(f"final CC / L / σ:    {history['cc'][-1]:.3f} / {history['L'][-1]:.2f} / {history['sigma'][-1]:.2f}")
    plot_results(g, agents, history, out_path='results.png')
