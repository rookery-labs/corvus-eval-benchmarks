# Specification: Expected Calibration Error (ECE) Methodology & Metrology

**Spec ID:** `SPEC-EVAL-001`  
**Status:** VALIDATED  
**Component:** `corvus-eval-benchmarks`

---

## 1. Theoretical Formulation

In autonomous agentic browsing, an agent must possess calibrated confidence estimates to reliably know when to execute an action via System 1 reflex or escalate to System 2 deliberate reasoning.

Given a dataset of $N$ samples $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, where:
- $y_i \in \{1, \dots, K\}$ is the ground-truth target action or element index.
- $\hat{y}_i = \arg\max_{k} p_{i,k}$ is the predicted class.
- $\hat{p}_i = \max_{k} p_{i,k} \in [0, 1]$ is the associated confidence score.

### 1.1 Confidence Binning
The confidence interval $(0, 1]$ is partitioned into $M$ disjoint bins:
$$B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right], \quad m \in \{1, \dots, M\}$$

### 1.2 Bin Statistics
For each non-empty bin $B_m$:
- **Bin Accuracy:**
  $$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$

- **Bin Confidence:**
  $$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

### 1.3 Expected Calibration Error (ECE)
The ECE is computed as the weighted average of the absolute difference between accuracy and confidence across all bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 1.4 Maximum Calibration Error (MCE)
$$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 2. Temperature Calibration Optimization

To calibrate raw model logits $Z \in \mathbb{R}^{N \times K}$:
$$p_{i,k}(T) = \frac{\exp(z_{i,k} / T)}{\sum_{j=1}^K \exp(z_{i,j} / T)}$$

The optimal scaling temperature $T^*$ is solved via bounded Nelder-Mead or L-BFGS-B optimization minimizing negative log-likelihood (NLL) and validated against ECE:
$$T^* = \arg\min_{T > 0} \mathcal{L}_{\text{NLL}}(Z / T, Y)$$

**Acceptance Criteria for System 1 Deployment:**
- Post-calibration $\text{ECE} \le 0.05$ (5.0%).
- Reliability diagram must exhibit monotonic increase with slope within $[0.90, 1.10]$.
