# How it works — concepts in plain language

## 1. The networking problem

A network is a set of **routers** joined by **links**. Each link has a **bandwidth** (how many bits
per second it can send), a **propagation delay** (how long a bit takes to travel along it) and an
**output queue** (a waiting line for packets that arrive faster than the link can send them).

A packet's **end-to-end delay** is the sum, over every hop, of

* **queueing delay** — time waiting in the output queue (this is what congestion inflates),
* **transmission delay** — packet size ÷ bandwidth,
* **propagation delay** — fixed by the link.

When a queue is full, newly arriving packets are thrown away (**tail-drop**). That is
**packet loss**. A link is **congested** when its queue is (nearly) full.

**Routing** decides which neighbour a router sends each packet to. Classic protocols such as OSPF
use the **shortest path** by a fixed cost. That is fast and stable, but the shortest path can be
congested while other paths sit idle.

## 2. The three routing methods compared

1. **Shortest Path (baseline 1).** Dijkstra with cost = propagation delay. Every packet between the
   same two routers always takes the same path, even if it is congested.
2. **Congestion-Aware heuristic (baseline 2).** Every 10 ms a "controller" recomputes Dijkstra with
   cost = propagation + current queueing delay + transmission time (+ a big penalty if the queue is
   ≥ 90 % full). It reacts to congestion, but only at the next update, and it moves *all* traffic
   for a destination at once, which can cause route flapping.
3. **DQN (our method).** At every hop, a neural network looks at the current state of each
   outgoing link (and one hop beyond) and picks the neighbour with the best learned long-term value.

## 3. Reinforcement learning in one minute

An **agent** observes a **state** *s*, takes an **action** *a*, receives a **reward** *r* and ends up
in a new state *s'*. It wants to maximise the total (discounted) future reward

    G = r₁ + γ·r₂ + γ²·r₃ + …        (γ = 0.98, a "discount factor")

The **Q-value** Q(s, a) is the expected G if we take action *a* in state *s* and act well afterwards.
If we know Q, the best action is simply `argmax_a Q(s, a)`.

**Q-learning** improves Q using the **Bellman equation**: the value of (s, a) should equal the
immediate reward plus the discounted value of the best next action:

    Q(s, a)  ←  r + γ · max_a' Q(s', a')

## 4. From Q-learning to a Deep Q-Network (DQN)

The state here has 65 numbers, so we cannot store a table of every state. Instead a small
**neural network** approximates Q: input = state, output = one Q-value per neighbour.

Three standard tricks make this stable:

* **Experience replay** — every hop is saved as `(s, a, r, s', mask', done)` in a buffer of
  100,000 transitions; training uses random mini-batches of 128, so consecutive, highly similar
  decisions don't dominate the learning.
* **Target network** — a slowly-updated copy of the network computes the target
  `r + γ·Q_target(s', a*)`. Without it the network would chase a target that moves every step.
  We update it softly: θ_target ← 0.01·θ + 0.99·θ_target each step.
* **Double DQN** — the online network *chooses* a* in s', the target network *evaluates* it. This
  reduces the over-estimation bias of plain max.

**Exploration:** ε-greedy. With probability ε we pick a random allowed neighbour, otherwise the best
one. ε falls linearly from 1.0 to 0.05 over the first 60 % of training.

**Loss:** Huber (smooth-L1) between Q(s, a) and the target; optimiser Adam, learning rate 5·10⁻⁴.

## 5. Our state, action and reward

**Decision point:** a packet is at router *u* and has destination *d*.

**Action:** "send to neighbour *k*" (*k* = 0…3, routers have at most 4 neighbours).

**State** (65 numbers, all scaled to roughly 0–1). For each neighbour slot:

| # | feature | why it helps |
|---|---|---|
| 0 | slot exists | routers have 2–4 neighbours |
| 1 | queue occupancy of link u→nb | direct congestion signal |
| 2 | estimated queue delay of u→nb | congestion in time units (depends on bandwidth) |
| 3 | propagation delay of u→nb | fixed cost of the hop |
| 4 | current bandwidth of u→nb | reveals degraded links |
| 5 | hops from nb to d | progress towards the destination |
| 6 | min propagation delay nb→d | remaining fixed delay |
| 7 | downstream queue delay at nb | congestion one hop further (SDN-style global view) |
| 8 | nb is the destination | final hop |
| 9 | recent drops on u→nb | loss history |
| 10 | recent utilisation of u→nb | how busy the link has been |

plus one-hot(current router), one-hot(destination), remaining TTL fraction.

**Action mask** (forbidden actions get Q = −∞): non-existent neighbours, routers this packet has
already visited (so routing loops are impossible), and neighbours from which *d* cannot be reached
within the remaining hop budget. If a packet ever reaches a dead end, revisits are allowed.

**Reward** for each hop, given when its outcome is known:

    r = 10·delivered − 0.1·hop_delay_ms − 5·dropped − 3·congestion_penalty
    congestion_penalty = max(0, occupancy − 0.7) / 0.3   (of the chosen queue, 0…1)

Adding up the hop rewards along a path gives `10 − 0.1 × end-to-end delay − …`, so maximising
reward means: deliver the packet, quickly, and avoid filling queues. The coefficients are the
illustrative ones from the project brief; they were not tuned extensively.

## 6. How a simulation step works (1 ms)

1. apply link events (e.g. link R4–R7 drops to 30 % bandwidth),
2. routers refresh their information (heuristic every 10 ms; DQN reads link state every ms),
3. packets that finished propagating arrive; those at their destination are delivered,
4. new packets are generated (Poisson arrivals per flow),
5. every waiting packet gets a next hop and joins that link's queue (or is dropped),
6. every link transmits as many whole packets as its bandwidth allows this ms,
7. statistics are recorded.

## 7. QoS metrics

| Metric | Definition |
|---|---|
| Average delay | mean end-to-end delay of **delivered** packets (dropped packets have no delay) |
| Jitter | per flow, mean |d_i − d_{i−1}| between consecutive delivered packets, averaged over flows |
| Throughput | delivered bytes × 8 ÷ traffic duration (Mbps) |
| PDR | delivered ÷ generated |
| Loss ratio | (queue-overflow drops + TTL drops) ÷ generated |
| Link utilisation | bytes sent ÷ bytes the link could have sent |
| Congestion frequency | fraction of (link, ms) samples with queue ≥ 80 % full |

Every generated packet ends as delivered, dropped, or still in flight when the run stops
(after a 300 ms drain with no new traffic).

## 8. Fair comparison

* All algorithms see the **same packet arrivals** for a given seed (common random numbers).
* Training, validation (model selection) and test scenarios use **disjoint random seeds**.
* An extra test uses **10 random traffic matrices** never used in training.
* Each result is the mean ± standard deviation over 5 seeds.

## 9. What the simulator does *not* model

TCP's own congestion control (senders here never slow down), routing-protocol message overhead
and delay of collecting link state, packet reordering effects on TCP, sub-millisecond timing,
hardware forwarding limits, link failures with detection delay, and real traffic traces.
Results are therefore comparisons *within this model*.
