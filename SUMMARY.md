# Multimodal, interpretable ML for pediatric appendicitis: a small reproducible study

**Question.** Using the public Regensburg pediatric appendicitis dataset (CC BY-NC 4.0, single centre), do frozen image features add to tabular clinical data for diagnosis, what accuracy is lost by routing predictions through human-readable concepts, and are the probabilities calibrated? This is a re-analysis with simple models, not novel research.

**Approach.** 578 patients with a diagnosis and at least one linkable ultrasound image (66% appendicitis). Stratified 60/20/20 patient-level split (seed 0); preprocessing fit on train only; tuning on validation only. Models: tabular logistic regression and gradient boosting; frozen ImageNet ResNet-50 embeddings (patient-level mean) with logistic regression; fused tabular + image; and a sequential concept bottleneck (11 clinical/ultrasound concepts predicted from image and labs; diagnosis from predicted concepts only). All metrics on the 116-patient test set with 95% patient-level bootstrap CIs; temperature scaling fit on validation only.

**Main results (test AUROC [95% CI]).**
- Tabular LR 0.963 [0.929, 0.988]; gradient boosting 0.969 [0.938, 0.991].
- Image-only 0.635 [0.520, 0.743]: weak.
- Fused 0.952 [0.909, 0.982]; fused minus tabular LR is -0.012 [-0.039, 0.009], so images added nothing detectable.
- Concept bottleneck 0.854 [0.771, 0.923]; fused minus bottleneck is 0.098 [0.032, 0.173].
- Temperature scaling did not improve test calibration for any model (fused ECE 0.084 [0.065, 0.151] before, 0.088 [0.065, 0.156] after).
- Sanity checks: no patient in more than one split; over 20 label permutations, mean test AUROC was 0.50-0.52 for four models (0.39 for the bottleneck, whose null is bimodal); single permuted runs varied widely.

**Honest limitations.**
- Small n (116 test patients), one split, one centre, no external validation.
- The cohort is the subset with linkable images and is enriched for appendicitis.
- Tabular inputs include the clinicians' own ultrasound findings and scores, and missingness patterns (such as an unmeasured appendix diameter) are among the top SHAP features, so the high tabular AUROC is not a screening result.
- The bottleneck's gap is confounded with its smaller input set and poorly predictable concepts (concept AUROCs 0.4-0.8); its probabilities are nearly constant and its coefficients are not clinically coherent, so it should not be read as "the cost of interpretability".
- Frozen ImageNet features only. Not clinical-grade.

**Next steps.** Test robustness without missing-indicator features; add an oracle (true-concept) bottleneck to separate concept-prediction error from concept-set insufficiency; try an ultrasound-specific encoder; repeated cross-validation; external validation.
