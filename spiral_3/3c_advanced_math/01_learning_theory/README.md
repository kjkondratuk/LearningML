# Module 01: Statistical Learning Theory

## Learning Objectives

- Understand the PAC learning framework and sample complexity bounds
- Compute VC dimension for common hypothesis classes
- Implement and verify concentration inequalities (Hoeffding, McDiarmid)
- Compute Rademacher complexity empirically

## Key References

- **Shalev-Shwartz & Ben-David.** "Understanding Machine Learning: From Theory to Algorithms." Ch. 3-6.
- **Boucheron, Lugosi & Massart.** "Concentration Inequalities: A Nonasymptotic Theory of Independence."

## Theoretical Background

### PAC Learning

A hypothesis class H is PAC learnable if there exists an algorithm that, for any epsilon
and delta, given m >= m_H(epsilon, delta) samples, returns h with:

    P(L_D(h) - min_{h' in H} L_D(h') > epsilon) < delta

### VC Dimension

The VC dimension of H is the largest set of points that H can shatter (realize all 2^d
labelings). For linear classifiers in R^d, VC(H) = d + 1.

### Sample Complexity

    m >= (1/epsilon) * (VC(H) * log(1/epsilon) + log(1/delta))

### Concentration Inequalities

**Hoeffding:** P(|mean(X_i) - E[X]| >= t) <= 2 exp(-2n t^2) for X_i in [a, b].

**McDiarmid:** If changing one input changes f by at most c_i, then
P(f - E[f] >= t) <= exp(-2t^2 / sum c_i^2).

> **Math Callout:** Concentration inequalities (Boucheron et al.), growth functions,
> Sauer's lemma. Resource: Shalev-Shwartz & Ben-David (free PDF online).

## Exercises

1. `vc_dimension_linear_classifier` -- Compute and verify VC dim = d+1
2. `pac_sample_complexity` -- PAC sample complexity bound
3. `rademacher_complexity_linear` -- Empirical Rademacher complexity
4. `hoeffding_bound` -- Hoeffding's inequality
5. `generalization_bound` -- VC generalization bound
6. `mcdiarmid_bound` -- McDiarmid's inequality

## Derivation Prompts

1. Prove Sauer's lemma: the growth function tau_H(m) <= (em/d)^d.
2. Derive the VC generalization bound from Hoeffding's inequality and the union bound.
3. Show that the Rademacher complexity of linear classifiers scales as O(1/sqrt(n)).
