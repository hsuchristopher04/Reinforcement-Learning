# Explainable CartPole: distilling a neural controller into a decision tree

**Project summary:** An eight-leaf decision tree imitated a neural reinforcement-learning teacher and completed 98.4% of 500 held-out CartPole episodes. The project investigates the failures as well as the improvement, then repeats the fitting procedure across five new demonstration datasets.

**Stack:** Python, PyTorch, Gymnasium, scikit-learn, NumPy, Matplotlib, HTML/CSS/JavaScript.

**Explore:** [demo landing page](index.html) · [quick start](README.md) · [reproduction guide](REPRODUCING.md).

## Problem and approach

A neural controller can balance the pole successfully without making its decisions easy to inspect. The question here was whether a small decision tree could retain strong control performance while exposing each action as a sequence of threshold decisions over four observations: cart position, cart velocity, pole angle, and angular velocity.

The pipeline trains a Double DQN teacher, collects its state-action demonstrations, and fits decision-tree students. Demonstrations are split by whole episode to keep trajectories out of both training and action-validation partitions simultaneously. Policy performance is measured through complete environment rollouts, not just agreement with the teacher on saved observations.

The work includes training and evaluation code, decision-tree export and inference, failure diagnosis, controlled comparisons, replication, and offline demonstrations with recorded decision-path highlighting. Bundled models and evidence carry source paths and SHA-256 hashes.

## Experimental progression

1. **Establish a successful teacher.** The saved neural teacher reaches the 500-step episode limit throughout the capacity study's held-out evaluation.
2. **Build an inspectable baseline.** A four-leaf tree retains high mean return but leaves many episodes incomplete. This makes completion rate a useful complement to average return.
3. **Test learner-state aggregation.** Collect states visited by the student and ask the teacher to label them. In this setup, aggregation substantially degraded the small tree; weighting did not produce a reliable improvement. These are retained negative results, not omitted attempts.
4. **Study capacity.** Compare leaf budgets 4, 8, 16, 32, and 64 on original and augmented data. Select the smallest actual tree meeting a pre-specified 90% validation completion threshold before evaluating on the test seeds.
5. **Check sensitivity to demonstration data.** Freeze the eight-leaf procedure, collect five new datasets from the fixed teacher, and report every fit without selecting a new winner.

## Results

Capacity study, 500 untouched matched test episodes:

| Policy | Mean return | Completion | Incomplete episodes |
| --- | ---: | ---: | ---: |
| Neural teacher | 500.00 | 100% | 0 |
| Saved four-leaf baseline | 485.66 | 70.8% | 146 |
| Selected eight-leaf tree | 498.84 | 98.4% | 8 |

The completion gain is **27.6 percentage points**. The eight-leaf policy is not uniformly better: its worst return was 288 versus 336 for the baseline. The baseline was depth-limited, whereas the capacity candidates used a leaf-budget growth strategy, so this comparison does not isolate leaf count as the sole cause of improvement. See [capacity protocol and results](CAPACITY.md).

Across five new demonstration datasets, completion rates were **90.8%, 99.0%, 98.4%, 97.8%, and 99.6%**, averaging **97.12%**. Three fitting seeds per dataset yielded identical policies within that dataset, leaving **five distinct policies across fifteen fits**. The evaluation seeds were shared across policies. The main replication units are the five demonstration collections, not fifteen independent model replications. See [replication protocol and results](REPLICATION.md).

## What the failures taught

High action agreement does not ensure that a student will remain in the states represented by its demonstrations. However, adding teacher-labeled learner states also does not guarantee improvement under a small model's constraints. The aggregation results motivated testing model capacity, while the non-monotonic capacity results showed that simply making a tree larger is not a reliable selection rule.

The replication policies' remaining failures were associated with the cart reaching the track boundary. The demos include both successful control and failure examples so a viewer can inspect behavior alongside the aggregate metrics. Playback uses saved trajectories; it is not a live browser simulation or an unbiased sample of episodes.

## Scope and limitations

These results concern one fixed teacher and default CartPole dynamics. They do not demonstrate robustness to sensor noise, altered physics, a broader initial-state distribution, or independent teacher-training seeds. Five demonstration collections provide evidence of repeatability within this setup, not a universal performance guarantee. Interpretability here means an inspectable rule path; no human interpretability study was performed.

## Two-minute walkthrough

1. Open the [capacity demo](demo-capacity/index.html). Explain the four observed quantities and that 500 steps counts as completion.
2. Compare the original and selected students on the same example seed. Follow the highlighted decision path.
3. Show a remaining failure. Explain why mean return alone can obscure incomplete episodes.
4. Return to the landing page's result table and replication figure. Distinguish the selected model's 98.4% result from the procedure's 97.12% mean across new datasets.
5. Close with the reproducible evaluation command and the limits of the evidence.

## Reusable portfolio copy

**Short description:** Distilled a Double DQN CartPole controller into an inspectable eight-leaf decision tree, achieving 98.4% completion on 500 held-out episodes. Investigated failed data-aggregation approaches and measured reliability across five new demonstration datasets.

**Résumé bullets:**

- Built a reinforcement-learning policy-distillation pipeline using PyTorch and scikit-learn; the selected eight-leaf student completed 98.4% of 500 held-out episodes versus 70.8% for the saved four-leaf baseline.
- Evaluated a fixed student-training procedure across five demonstration datasets, reporting 90.8–99.6% completion and documenting aggregation failures, reproducible artifacts, and interactive decision-path playback.

**Interview opening:** “I wanted to see how much of a successful neural controller could be captured by a small decision tree. The interesting part was that high average reward hid failures, and collecting more learner-state data made the small tree worse. I tested capacity separately, selected an eight-leaf policy using validation rollouts, and then checked how its training procedure behaved with five new demonstration datasets.”
