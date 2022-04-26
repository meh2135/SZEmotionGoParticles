# SZEmotionGoParticles

Library and notebooks for schizophrenia-induced emotion mismapping analysis.

## Overview

* Sum the squares of the fine grained emotions for each coarse emotion:
  * $\tilde{p}_{coarseEmotion,trialNumber}:=\sum_{fineEmotion \in coarseEmotion} s_{fineEmotion, trialNumber}^{2}$
* Normalize the coarse emotions so they sum to 1 for each *trial*
  * $p_{coarseEmotion,trialNumber} = \tilde{p}_{coarseEmotion,trialNumber} / \sum_{coarseEmotion^{\prime}} \tilde{p}_{coarseEmotion^{\prime},trialNumber}$
* From here on I'll use:
  * $i$ subject
  * $j$ stimulus
  * $k$ coarse emotion
  * $n_{e}=4$ the number of emotions
  * $n_{s}=14$ the number of stimuli
  * $n_{p}=?$ the number of subjects

### Model

* For each stimulus $j$ we have a vector $r_{j}$ of length $n_{e}$ which is a normative distribution over emotions for that stimulus (elements a between 0 and 1 and sum to 1)
* skipping details and *grotesquely* abusing notations, define $\mathscr{N}_{i}\left(x \right)$ to be a noised representation of the distribution over emotions $x$ that includes both a trial specific pure noise component and a subject level random effect
* $\beta$ is an $n_{e}$ by $n_{e}$ matrix representing how much of each emotion is *piped* to each other emotion. Each row (column??) is a distribution
* We observe
  * $\mathscr{N}_{i}\left(r_{j} \right)$ for HCs
  * $\mathscr{N}_{i}\left(\beta^{\prime} r_{j} \right)$ for SZs

### Particle analogy

Imagine that each stimulus emits par
