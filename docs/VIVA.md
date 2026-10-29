# Viva preparation — questions and answers

Answers are short so you can say them out loud. Numbers quoted come from `results/RESULTS.md`;
re-check them if you retrain.

## A. Problem and motivation

**1. What problem does your project solve?**
Choosing routes that keep delay and loss low when traffic changes. Static shortest-path routing
sends all traffic for a destination along one path even when it is congested and alternatives are
idle. We train a DQN agent that picks each packet's next hop from the current network state.

**2. Why not just use shortest-path routing?**
Its link costs are fixed, so it cannot react to load. In our hotspot traffic the shortest paths
share the R1–R4–R7 core; at load 0.75 and above those links saturate and shortest path loses a
large share of packets while side paths are lightly used.

**3. What does "congestion-free" mean in your title?**
It is the optimisation goal: minimise congestion. If offered traffic exceeds total capacity, every
algorithm must drop packets — no routing method can guarantee zero congestion.

**4. What is QoS?**
Quality of Service: measurable properties of delivery — delay, jitter, throughput, packet loss /
delivery ratio. We measure all of them.

## B. Networking concepts

**5. What are the components of end-to-end delay?**
Queueing + transmission (size/bandwidth) + propagation (distance), per hop. (Processing delay is
not modelled.) Congestion mainly increases queueing delay.

**6. How are packets lost in your simulator?**
Tail-drop when a 30-packet output queue is full, or TTL expiry after 12 hops.

**7. What is jitter and how do you compute it?**
Variation of delay between consecutive packets of a flow. Per flow, mean |dᵢ − dᵢ₋₁| over
consecutive delivered packets, averaged over flows (the idea of RFC 3550 without smoothing).

**8. Throughput vs. PDR?**
Throughput is delivered bits per second (Mbps). PDR is the fraction of generated packets that were
delivered. When nothing is lost, throughput equals offered load, so throughput only separates
algorithms once loss begins.

**9. Why is average delay computed only for delivered packets?**
A dropped packet has no arrival time. Its effect appears in loss/PDR instead. That is why you must
read delay together with loss: an algorithm that drops slow packets can look artificially fast.

**10. How does this relate to SDN?**
In SDN a central controller has a global view of link state and installs forwarding rules. Our
congestion-aware heuristic behaves like a controller polling every 10 ms; the DQN's
"downstream delay" feature also assumes the controller shares one-hop-ahead state. A real SDN
deployment (e.g. Mininet + Ryu/ONOS) would run the policy in the controller.

## C. Reinforcement learning

**11. What are the state, action and reward?**
State: 65 numbers — for each of 4 neighbour slots: queue occupancy, queue delay, propagation delay,
bandwidth, hops and delay to destination from that neighbour, downstream queue delay, is-destination
flag, recent drops, recent utilisation; plus one-hot current router, one-hot destination,
remaining TTL. Action: which neighbour. Reward per hop:
`10·delivered − 0.1·hop_delay − 5·dropped − 3·congestion_penalty`.

**12. Why per-hop decisions instead of choosing a whole path?**
The action space stays tiny (≤ 4), one network serves every router and destination, and each hop
uses the freshest local information. A whole-path action would need one output per path and
re-training for every source/destination pair.

**13. What is the Q-value here?**
The expected discounted future reward of sending the packet to a given neighbour — roughly
"10 minus 0.1 × the remaining delay, minus penalties for drops and congestion".

**14. Why experience replay?**
Consecutive decisions are highly correlated (same queues, same flows). Random mini-batches from a
100k buffer de-correlate the data and reuse each experience many times.

**15. Why a target network?**
The target `r + γ max Q(s', ·)` depends on the network being trained. Using a slowly updated copy
keeps the target stable. We use soft updates with τ = 0.01.

**16. What is Double DQN and why use it?**
Plain DQN uses max over noisy estimates, which over-estimates values. Double DQN chooses the next
action with the online network and evaluates it with the target network.

**17. How do you prevent routing loops?**
Action masking: a packet cannot be sent to a router it has already visited, and not to a neighbour
from which the destination is unreachable within its remaining TTL. Masked actions get Q = −∞. The
TTL (12) is a second safety net. A unit test checks DQN paths are loop-free.

**18. How does exploration work?**
ε-greedy: random allowed neighbour with probability ε, decaying linearly 1.0 → 0.05 over the first
60 % of training ticks.

**19. Why Huber loss?**
It is quadratic for small errors and linear for large ones, so occasional large targets (e.g. a
drop after a long wait) don't produce huge gradients.

**20. What is γ and why 0.98?**
The discount factor. Paths are short (≤ ~6 hops), so γ close to 1 keeps the delivery reward
visible from the first hop while still bounding returns.

**21. Why these reward coefficients?**
They come from the project brief as a starting point (10, 0.1, 5, 3). Summed over a path, the
delay terms equal −0.1 × end-to-end delay, so the return directly reflects QoS. They were not
extensively tuned — that is listed as future work.

**22. Is it multi-agent?**
All routers share one Q-network (parameter sharing) — a "single shared policy" view. Each router
uses only its own local view plus one-hop downstream information.

**23. Why is the neural network written in NumPy?**
It keeps the project dependency-light and lets me show the backpropagation explicitly
(`qnet_numpy.py`), verified by a finite-difference gradient-check test. An equivalent PyTorch
backend is included (`--backend torch`) and a test checks the two give the same outputs.

## D. Experiment design

**24. How did you make the comparison fair?**
Same topology, same traffic per seed for all algorithms (common random numbers), identical metric
code, 5 test seeds per point, mean ± std reported.

**25. How are training and testing separated?**
Training uses random scenarios from one seed stream; the best model is chosen on separate
validation seeds; final results use test seeds 900000+; plus 10 random traffic matrices never seen
in training.

**26. What scenarios did you test?**
Load sweep 0.25–2.0 on the hotspot matrix; named light (0.4), moderate (0.9), heavy (1.6); dynamic
(load 0.4 → 1.0 → 1.6 → 0.8 with link R4–R7 at 30 % bandwidth for a third of the run); unseen
random traffic matrices; and a heuristic update-interval sensitivity test.

**27. Why is the heuristic baseline so strong?**
It has a global view refreshed every 10 ms, which is optimistic for real networks. We kept it
strong deliberately, and show in the sensitivity test how it changes with 50/100 ms updates.

## E. Results (check against results/RESULTS.md)

**28. What did you find?** — see the *Results summary* section at the end of this file.

**29. Does DQN always win?**
No. Report exactly what the tables show, including where the heuristic matches or beats it. At
light load all methods are similar because there is no congestion to avoid.

**30. Why does DQN have more hops on average?**
It spreads traffic over longer but less loaded paths. That costs some propagation delay but avoids
much larger queueing delay.

## F. Limitations and future work

**31. Main limitations?**
Simulation only; no TCP reaction to loss; link state is assumed known instantly (DQN) or every
10 ms (heuristic) with no signalling cost; one small topology — the network's input size depends on
the number of routers, so a new topology needs retraining; per-packet routing can reorder packets
of a flow; reward weights not tuned; 300 training episodes.

**32. Could this run on real routers?**
Not directly. Practical steps: run the trained policy in an SDN controller (Ryu/ONOS) on Mininet,
make decisions per flow or per flow-batch instead of per packet, collect link statistics via
OpenFlow port counters, then measure with iperf/ping.

**33. How would you scale to bigger networks?**
Use topology-independent inputs (e.g. a Graph Neural Network over link features) instead of one-hot
router IDs; train on many topologies.

**34. How would you reduce packet reordering?**
Route per flow (or flowlet) instead of per packet: pick a path when a flow/flowlet starts and keep
it while it is active.

**35. What would you improve first?**
Reward-weight tuning with a validation sweep, prioritized replay, flowlet-level decisions, and a
Mininet experiment.

## G. Code questions

**36. Walk through one simulation tick.**
Events → router on_tick → arrivals/deliveries → new packets → batched next-hop decisions
(enqueue or drop) → links transmit using byte credits → statistics. (`simulator.py::step`)

**37. Where is the reward computed?**
`qosroute/routing/dqn_router.py`: on arrival at the next router (`choose`), on delivery
(`on_deliver`) and on drop (`on_drop`).

**38. How do you know the simulator is correct?**
Unit tests: packet conservation (generated = delivered + dropped + in flight), single-packet delay
equals the sum of propagation delays, delays never below the minimum propagation delay, queues
never exceed capacity, overflow produces drops, shortest path is static, gradient check of the
network.

**39. What is the byte-credit scheduler?**
Each ms a busy link gains credits equal to its bandwidth in bytes and sends whole packets from the
head of its queue while credits ≥ packet size. This models bandwidth with variable packet sizes
at 1 ms resolution.

**40. What is saved in the model file?**
`models/dqn_router.npz`: the six weight/bias arrays of the online network and a JSON metadata
string (state size, number of actions, hyper-parameters, training episode, validation score,
topology edges).

---

## Results summary (from results/RESULTS.md — 5 test seeds, mean values)

* **Shortest path** loses 16 % of packets at load 0.75, 31 % at 1.0, 53 % at 2.0. Its fixed paths
  overlap on R1→R4, which saturates.
* **Up to capacity (load ≤ 1.25):** heuristic and DQN both deliver 100 %. DQN delay is lower
  (load 1.0: 10.0 vs 13.3 ms; load 1.25: 11.3 vs 18.0 ms) and jitter is 1.5–5.5× lower
  (load 1.0: 1.4 vs 4.5 ms).
* **Overload (load ≥ 1.5):** heuristic delivers 0.6–1.2 percentage points more (load 1.5: 98.6 %
  vs 97.4 %); DQN has 21–32 % lower delay and 3.5–6× lower jitter.
* **Dynamic scenario:** delay SP 37.7 / heuristic 24.8 / DQN 20.3 ms; PDR 0.561 / 0.985 / 0.982.
* **Unseen traffic matrices:** both adaptive methods 100 % delivery; DQN lower delay in 15 of 20
  paired cases (small margin, ~0.2–0.5 ms on average).
* **Heuristic with slower updates (50/100 ms):** heavy-scenario PDR drops from 0.947 to
  0.706/0.639 — its advantage depends on very fresh global state.

**Q: Your DQN loses more packets than the heuristic under overload — so is it worse?**
Under overload it trades ~1 percentage point of delivery for 21–32 % lower delay and much lower
jitter. Which is "better" depends on the application: delay-sensitive traffic (voice, video calls)
prefers the DQN's trade-off; bulk transfer prefers fewer drops. The reward weights set this
trade-off; raising the drop penalty would push the agent towards fewer drops.

**Q: Why does shortest path show *lower* delay than the heuristic at load 2.0?**
Delay is averaged only over delivered packets, and shortest path drops 53 % of its packets —
mostly those that would have waited longest. Always read delay together with PDR.

**Q: Why did you pick the episode-60 model, not the last one?**
Model selection by validation score (separate seeds). The score plateaued after episode 60
(7.76–7.89); with only 6 validation runs, later differences are within noise.

**Q: Why does the training reward curve look so noisy?**
Each episode has a random load (0.3–2.0) and possibly a degraded link; overloaded episodes have
low reward whatever the policy. The validation curve (fixed scenarios, greedy policy) is the
cleaner signal.
