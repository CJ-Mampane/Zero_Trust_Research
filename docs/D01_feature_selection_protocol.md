# D01 — Feature-selection protocol FS-1

**Status:** FROZEN on 2026-10-11, *before* `5_feature_selection.py` was first run.
**Machine-readable version:** `config/feature_selection_protocol.json`. The script reads every parameter from that file. On its first run it records the file's SHA-256 fingerprint in `results/05_feature_selection/protocol_lock.json`, and on every later run it refuses to continue if the file has changed.
**Approved by:** the student, 2026-10-11. The supervisor is to be informed.
**Data used:** development partitions only. The 30% test partitions are not opened.

---

## 1. The decision

Every feature-selection decision for IoT-23 (which variables, how many) is driven by **macro-F1 on validation rows from captures that the fold's training data never contained**. In design B these are the rows with `scenario_group == whole_scenario`.

Rows from CTU-IoT-Malware-Capture-35-1 are always scored and reported. They never drive a decision.

For RT-IoT2022, which has no capture, session or device identifier, decisions use all validation rows of the stratified 5-fold design.

The scores of the two measuring instruments (Random Forest and XGBoost) are averaged per fold. Neither instrument is allowed to decide alone.

## 2. Why this criterion

1. **The purpose of a Zero Trust trust model is to judge traffic it has not seen before.** A development criterion should estimate that, not how well the model recognises more traffic from a capture it trained on.

2. **Random validation overestimates performance when observations are grouped.** In data with group structure, ordinary random cross-validation seriously underestimates prediction error. Splitting by block or group is recommended whenever such dependence exists, especially when the goal is predicting new data or selecting predictors (Roberts et al., 2017). This is a general statistical principle established in ecology, not an IoT-23-specific finding. Flows from one IoT-23 capture share a device, a malware campaign and a time window, so they are a grouped structure of exactly this kind.

3. **The project's own development data shows the problem.**
   - Capture-35-1 is 86.5% of development rows.
   - Under the historical row-level CV, the feature-count curve was flat at about 0.994 macro-F1. Candidates differed by about 0.0001 (verified earlier from the historical curve file).
   - A criterion that cannot tell candidates apart turns the selection into noise.
   - This script also reports Capture-35-1 and all-row scores, so the size of the difference is measured on development data, not assumed.

4. **Security machine learning has known pitfalls of exactly this type.**
   - Arp et al. (2022) identify common pitfalls in the design and evaluation of learning-based security systems. They show these pitfalls can produce unrealistic performance and misleading interpretations.
   - D'hooge et al. (2022) show that destination port acts as a shortcut feature in machine-learned intrusion detection models.
   - Choosing features on unseen-capture rows makes it harder for capture-specific shortcuts to win the selection.

## 3. Other design elements and their justification

| Element | Choice | Justification |
|---|---|---|
| Unit of selection | Original variables; one-hot columns of a variable kept or removed together | Permuting one dummy at a time spreads a categorical variable's importance across its dummies (historical limitation b). Grouped permutation importance is the established extension (Gregorutti et al., 2015). It also keeps the result interpretable for SRQ2. |
| Ranking measure | Permutation importance on held-out rows | Model-agnostic. Avoids the known bias of impurity importance toward variables with many categories or values (Strobl et al., 2007). |
| Nesting | Ranking for outer fold f uses only the other folds | Selection outside the CV loop is optimistically biased (Ambroise & McLachlan, 2002; Cawley & Talbot, 2010). This removes historical limitation (a). |
| Unsupervised filters fitted on all development rows | Constant, rare-level and correlation filters | They do not use labels. Unsupervised screening before cross-validation does not bias the error estimate the way supervised screening does (Hastie, Tibshirani & Friedman, ESL §7.10.2). |
| Rare-level rule | Keep a level if p(1−p) > 0.01 in all development rows **or** in non-Capture-35-1 rows | Capture-35-1 makes up 86.5% of rows, so a level common in the other captures could fall under 1% overall. It would then be deleted purely because one capture is large. The 0.01 value is a stated convention (historical value); the "or" rule is a researcher decision and is labelled as such. |
| Correlation filter | Spearman, \|ρ\| > 0.90, findCorrelation tie-break; port excluded | Byte and packet counts are extremely skewed, so Pearson is driven by outliers while rank correlation is robust. The tie-break replaces the historical column-order rule, which was arbitrary. 0.90 is a convention (historical value). |
| Count rule | One-standard-error rule | Prefers the simplest model statistically indistinguishable from the best (Hastie et al., ESL §7.10). |
| Folds | IoT-23 K = 3 (design B), RT K = 5 | K = 3 is imposed by the data: only three development captures carry substantial Suspicious traffic. |
| Class codes | Fixed: Trusted = 0, Suspicious = 1, Untrusted = 2 | Removes the historical confusion between alphabetical codes and the ordinal order. |
| Stability | Nogueira stability index and Kendall τ across nested rankings; mutual-information ranking as a cross-check | A selection that changes when the data change cannot be trusted (Nogueira, Sechidis & Brown, 2018). Reported only; never used for decisions. |

## 4. Alternatives considered and rejected

- **(i) All validation rows.** Mirrors the test mix (75% Capture-35-1), but cannot discriminate between candidates (see §2.3).
- **(iii) Equal-weight average of the two parts.** Still lets the non-discriminating part dilute the signal, and adds a weighting choice with no principled value.
- **Fully grouped CV including Capture-35-1 as one group.** Infeasible: one fold would hold 86.5% of the data.
- **Wrapper methods (RFE, RFECV, forward or backward search).** Too costly at 1.49M rows on 8 GB. They also overfit the validation procedure (Cawley & Talbot, 2010).

## 5. Limitations (to be reported in the thesis)

1. **Test-informed design.** Criterion (ii) and validation design B were adopted after the historical pipeline's test results showed a gap between Capture-35-1 and unseen captures. The general literature (§2) supports them independently, but their adoption cannot be called blind. The new test evaluation is therefore not an independent confirmation of the design.
2. **Few captures, high variance.**
   - Unseen-capture Suspicious performance rests on three development captures (48-1, 1-1, 49-1).
   - Fold sizes are uneven: 14.5k, 144k and 42k selection rows.
   - Fold scores will differ a lot, and a standard error from K = 3 folds is itself uncertain. No unbiased estimator of k-fold CV variance exists (Bengio & Grandvalet, 2004).
   - The selected count must be described as "where the predeclared rule landed", not as a sharp optimum.
3. **Partial grouping only.** Capture-35-1 rows are still split row-wise across folds. The criterion keeps them out of decisions, but the final models train on them.
4. **Generalisation scope.** Generalising to unseen captures *within IoT-23* is not the same as generalising to other networks, tools or datasets (cross-dataset validation).
5. **Unit of analysis.** Rows are distinct flow records, not traffic volume, and labels are per flow, not per device. Macro-F1 weights the three classes equally, whatever their prevalence.
6. **Permutation importance with correlated variables.** Correlated variables share credit, and permuting them creates unrealistic combinations of values. The 0.90 filter removes only the strongest cases.
7. **Measuring instruments.** Ranking uses fixed, untuned RF and XGBoost (200 trees). Tuned models could rank variables differently.
8. **Nested rankings use less data.** With K = 3, each nested ranking for IoT-23 is trained on a single fold, so nested rankings are noisier than the final one.
9. **What the curve estimates.** It estimates the performance of the selection *procedure*, not of the exact final variable list.
10. **RT-IoT2022.** No grouping is possible, so its development scores measure in-distribution performance on distinct flow records only.
11. **Destination port** stays a candidate here. Whether to keep it is decided by a later, separately predeclared ablation (D'hooge et al., 2022).

## 6. Changes from the historical `4_feature_selection.py`

1. Grouped variables instead of single dummies.
2. Nested ranking.
3. Spearman with findCorrelation tie-break instead of Pearson with column-order tie-break.
4. Rare-level "or" rule.
5. Criterion (ii) on design-B folds instead of row-level folds.
6. RT-IoT2022 uses 5 folds instead of 3.
7. Fixed class codes.
8. Preprocessing specification saved for later use on test.
9. **The test set is not transformed here**; that happens only in the final evaluation script.
10. Checkpointing and stability diagnostics added.

## 7. References

Verified during this session: D'hooge 2022, Roberts 2017, Arp 2022, Nogueira 2018, Gregorutti 2015.
Cited from memory: the rest. Check each DOI before final submission.

- Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS*, 99(10), 6562–6566. https://doi.org/10.1073/pnas.102102699
- Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., & Rieck, K. (2022). Dos and Don'ts of Machine Learning in Computer Security. *31st USENIX Security Symposium*, 3971–3988. https://www.usenix.org/conference/usenixsecurity22/presentation/arp
- Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR*, 5, 1089–1105.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107.
- D'hooge, L., Verkerken, M., Volckaert, B., Wauters, T., & De Turck, F. (2022). Establishing the Contaminating Effect of Metadata Feature Inclusion in Machine-Learned Network Intrusion Detection Models. *DIMVA 2022*, LNCS 13358, 23–41. https://doi.org/10.1007/978-3-031-09484-2_2
- Gregorutti, B., Michel, B., & Saint-Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis. *Computational Statistics & Data Analysis*, 90, 15–35. https://doi.org/10.1016/j.csda.2015.04.002
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §7.10. Springer.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer (findCorrelation procedure, ch. 3).
- Nogueira, S., Sechidis, K., & Brown, G. (2018). On the Stability of Feature Selection Algorithms. *JMLR*, 18(174), 1–54. https://jmlr.org/papers/v18/17-514.html
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25. https://doi.org/10.1186/1471-2105-8-25
