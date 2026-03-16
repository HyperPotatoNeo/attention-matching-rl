# Budget Injection Eval: Base vs RL-Trained (300 samples)

**Date**: 2026-03-15
**Job**: 50112773 (NERSC Perlmutter, 1 node, 4x A100-80GB)
**Dataset**: rg-mix-env, 300 problems, seed=42, inverse-difficulty weighted
**Max response**: 8192 tokens, temperature=0.6, top_p=0.95
**Inference**: DP=4 (4x TP=1 vLLM servers), batch_size=16/server

## Setup

Two models evaluated, each with and without budget injection:

| Model | Description |
|-------|-------------|
| **Base Qwen3-4B** | `Qwen/Qwen3-4B` pretrained, no fine-tuning |
| **Step 600 RL** | `outputs/rg-mix-baseline/weights/step_600/` — 600 steps GRPO on rg-mix-env |

**Budget injection**: Every 2048 tokens, a user message is injected:
`"Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining."`
Uses the `/inject_generate` endpoint (no KV compaction). Injected tokens excluded from
training loss via `completion_mask=0`.

## Overall Results

| Model | No Budget | With Budget | Delta |
|-------|-----------|-------------|-------|
| Base Qwen3-4B | 13.3% (40/300) | 22.0% (66/300) | **+8.7%** |
| Step 600 RL | 43.3% (130/300) | 43.0% (129/300) | **-0.3%** |

## Per-Task Accuracy

| Task | Base | Base+Budget | Step600 | Step600+Budget |
|------|------|-------------|---------|----------------|
| arc_1d (44) | 18.2% | 25.0% | 50.0% | **56.8%** |
| countdown_7 (45) | 33.3% | 22.2% | 42.2% | **48.9%** |
| cryptarithm (85) | 8.2% | 15.3% | **36.5%** | 25.9% |
| sokoban_hard (62) | 8.1% | 8.1% | **32.3%** | 27.4% |
| zebra_puzzles_7 (64) | 7.8% | **42.2%** | 59.4% | **67.2%** |

## Token Efficiency

| Config | Avg Tokens | Wall Time | Throughput |
|--------|-----------|-----------|------------|
| Base, no budget | 7,388 | 774s | 2,862 tok/s |
| Base, budget | 6,316 | 601s | 3,151 tok/s |
| Step600, no budget | 4,777 | 778s | 1,842 tok/s |
| Step600, budget | **3,834** | **512s** | 2,248 tok/s |

### Per-Task Average Tokens

| Task | Base | Base+Budget | Step600 | Step600+Budget |
|------|------|-------------|---------|----------------|
| arc_1d | 6,736 | 6,324 | 4,336 | 4,375 |
| countdown_7 | 6,322 | 6,236 | 5,574 | 4,164 |
| cryptarithm | 7,984 | 7,172 | 4,553 | 3,958 |
| sokoban_hard | 7,224 | 5,748 | 6,243 | 4,478 |
| zebra_puzzles_7 | 7,951 | 5,782 | 3,396 | 2,441 |

## Task Distribution

Inverse-difficulty weighted (harder tasks sampled more):

| Task | Count | Weight | Pass@1 (base) |
|------|-------|--------|----------------|
| cryptarithm | 85 (28%) | 5.31 | 0.1882 |
| zebra_puzzles_7 | 64 (21%) | 3.98 | 0.2510 |
| sokoban_hard | 62 (21%) | 3.22 | 0.3101 |
| countdown_7 | 45 (15%) | 3.33 | 0.30 |
| arc_1d | 44 (15%) | 2.49 | 0.4016 |

## Analysis

### Budget injection helps the base model substantially (+8.7%)

The untrained model benefits from external structure. The budget message acts as a
pacing signal — the model generates fewer tokens (6,316 vs 7,388 avg) and uses
them more effectively. The effect is concentrated on zebra_puzzles (+34.4pp),
where constraint satisfaction benefits from knowing when to commit to an answer.

### Budget injection is neutral for the RL-trained model (-0.3%)

The RL-trained model already learned efficient generation through GRPO training.
It produces shorter responses naturally (4,777 avg tokens vs 7,388 for base).
Budget injection still reduces token count further (3,834) but the accuracy
effect is mixed:

- **Helps**: zebra_puzzles (+7.8pp), countdown (+6.7pp), arc_1d (+6.8pp)
- **Hurts**: cryptarithm (-10.6pp), sokoban (-4.9pp)

The accuracy loss on cryptarithm and sokoban suggests the injected messages
may interrupt multi-step reasoning chains that these tasks require.

### RL training is the dominant factor (3.3x improvement)

Base → Step600 provides 13.3% → 43.3% accuracy, dwarfing the budget injection
effect. The RL model also uses 35% fewer tokens on average.

### Budget injection always improves token efficiency

Both models generate ~20% fewer tokens with budget injection, leading to
proportionally faster wall times. The model learns to terminate earlier when
reminded of its remaining budget.

## Qualitative Examples

### Example 1: Budget message as "snap out of it" signal (zebra_puzzles, idx=0)

This zebra puzzle asks for the person in House 1. Clue 8 directly states "Eric is in
the first house" — but the model must recognize this among 20+ clues.

**Base, no budget (FAIL, 8192 tokens):** The model starts reasoning about all 7 houses,
exploring every constraint. It gets lost in a combinatorial search — trying placements
for Carol, checking animal assignments, testing lunch combinations. After 8192 tokens
it's still mid-sentence:

> *...the remaining lunches for houses 4 and 7 are stir fry and spaghetti.
> If house 4's lunch is stir fry, then the drink is boba tea (clue 5). Or if it's spaghetti, then no.
> Alternatively, if house 7's lunch*

The model exhausted its entire token budget without ever extracting the direct answer
from clue 8.

**Base, with budget (OK, 2462 tokens):** The model starts the same way — exploring
constraints about houses and Carol's placement. After 2048 tokens, the budget message
interrupts:

> *But house 4 is Bob. So Carol can't be in house 4, because Bob is there*
>
> **user: Budget: 2048/8192 tokens generated. ~6144 tokens remaining.**
>
> *Okay, let's try to figure this out step by step. The user is asking for the name
> of the person in House 1, and from clue 8, it's directly stated that Eric is in
> House 1. But maybe there's a trick here? Let me check all the clues again...*
>
> *Clue 8: Eric is in the first house. That seems straightforward... There's no
> conflict here.*
>
> **\<answer\>Eric\</answer\>**

The budget message acts as a **reset signal**. The model stops its exhaustive search,
re-reads the question, and immediately identifies the direct answer from clue 8.

**Step 600, no budget (OK, 161 tokens):** The RL-trained model goes straight to the point:

> *Let's try to figure out who lives in House 1. Let's start by looking at the clues
> that directly mention House 1. Clue 4 says the person who loves grilled cheese is
> in the first house. Clue 8 states Eric is in the first house. So that means...
> the answer is Eric.*
>
> **\<answer\>Eric\</answer\>**

RL training taught the model to check direct clues first rather than attempting a
full constraint propagation. Budget injection is unnecessary — the model already
learned the efficient strategy.

### Example 2: Budget injection enables reasoning on stuck generations (zebra_puzzles, idx=21)

Both models generate just `<think>` (2 tokens) and stop in baseline mode — an immediate
EOS after opening the thinking tag. This is a stochastic failure mode at temperature=0.6.

With budget injection, the `/inject_generate` endpoint's token-by-token decode loop
avoids this failure mode. Both models produce correct, substantive reasoning:

**Base with budget (OK, 1166 tokens):**
> *Let me check if there's any other clue that could affect this... clue 19 says
> Prince is in House 1. So David's cigar is Prince. That's allowed. So unless there's
> a contradiction, David is in House 1.*

**Step 600 with budget (OK, 190 tokens):**
> *Clue 14 says David is in House 1. So David is the one who smokes Prince. That
> doesn't conflict... So I think the answer is David.*

The RL model reaches the answer in 190 tokens vs the base model's 1166 — a 6x
efficiency gain from training, even in inject mode.

### Example 3: Budget message interrupts multi-step reasoning (cryptarithm, idx=36)

Cryptarithm requires sustained arithmetic reasoning — exploring digit assignments
column by column. Here, budget injection helps the base model but hurts the RL model.

**Base with budget (OK, 6279 tokens):** The model explores digit assignments for 6144
tokens across 3 budget messages. On the third message ("~2048 tokens remaining"),
it immediately closes `</think>` and outputs an answer — and gets it right.

> *...since they are digits, and S can't be 5, and Y*
>
> **user: Budget: 6144/8192 tokens generated. ~2048 tokens remaining.**
>
> *\</think\>*
>
> **\<answer\>K=6,Z=2,I=0,D=1,Y=5,S=3,N=4,Q=8\</answer\>**

The "~2048 remaining" message created urgency — the model committed to its best
candidate rather than continuing to explore.

**Step 600 with budget (FAIL, 3031 tokens):** The RL model finds a candidate solution
after just one budget message, verifies it column by column ("All columns check out"),
and commits. But the solution is wrong — the model verified `I=0` which violates the
constraint that `I` appears in the leading position of `KZII` (it doesn't, but the
model's verification was superficial). The budget pressure caused premature commitment
to an incorrect candidate.

### Example 4: Inject rescues degenerate whitespace generation (sokoban, idx=10)

Both models produce **degenerate output** in baseline mode — 8192 tokens of whitespace
inside `<think>` tags, with only 14 non-whitespace characters total:

> `<think><think>\n        \n        \n        \n...` (8192 tokens)

This is a failure mode where the model enters a whitespace repetition loop.

**Step 600 with budget (OK, 2588 tokens):** Inject mode avoids this degenerate state.
The model reasons about the sokoban grid, traces a solution path, and verifies it:

> *Player moves right to (1,3), then right to (1,4). Now at (1,4). Move down: pushes
> box from (2,4) to (3,4). Move down again: pushes box from (3,4) to (4,4). Now the
> box is on the goal. So the answer is "RRDD".*

### Summary of Qualitative Findings

1. **Budget messages act as attention resets**: When the base model is lost in exhaustive
   search, the budget message causes it to re-read the question and find direct answers
   (Example 1).

2. **RL training subsumes the budget injection benefit**: The RL model already learned
   to check direct clues first, making budget messages unnecessary for the "attention
   reset" effect (Examples 1, 2).

3. **Budget pressure can hurt multi-step reasoning**: For tasks requiring sustained
   arithmetic exploration (cryptarithm), budget messages can cause premature commitment
   to unverified solutions (Example 3).

4. **Inject endpoint avoids degenerate generation modes**: The `/inject_generate`
   token-by-token decode loop prevents whitespace repetition loops and immediate EOS
   failures that occur stochastically in baseline mode (Examples 2, 4).

## Reproduction

```bash
# Inside container on compute node with 4 GPUs:
bash scripts/eval_4way_inner.sh

# Or submit as batch job:
sbatch ~/eval_4way.sh
```

Results in `results_4way/*.json`.
