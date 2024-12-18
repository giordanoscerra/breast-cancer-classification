**DATASET SOURCE**: 

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183947

**FEATURE SELECTION**:

This was essential to reduce computational costs. The analysis without feature selection yields a score around 0.5, when we can narrow down the scope of our study and achieve much better results near perfection.
- p_values threshold (tried: 0.05, 0.01) -> more accuracy with 0.01 and fewer features
- either significant or non-significant genes can be used

**HYPERPARAMETERS**:
- FOR RANDOM FOREST CLASSIFIER: n_estimators (tried up to 100)
- FOR SVM: kernel (rbf or linear)

**INSTRUCTIONS**:

In the second cell we can choose these hyperparameters:

- `p_values_threshold = 0.01`

- `features_kind = "significant_genes"`

- `n_estimators = 100`

- `kernel= "linear"`

Let's change them a little and see how the recaps at the end change. Remember to put `p_values_threshold=0.05` when `features_kind = "non_significant_genes"`, so that we narrow down the features scope.

The goal is to see which genes work best to discriminate cancer from cancer free tissues. Those can be interesting biomarkers. We can then investigate if they're upregulated or downregulated in our dataset and understand why and the consequences by studying the specific gene role.


