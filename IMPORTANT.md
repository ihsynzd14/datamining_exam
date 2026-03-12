Build `datamining part 1/notebooks/03_classification_regression.ipynb` from scratch.

---

## GUIDELINE COMPLIANCE CHECKLIST (100% required — do not skip any item)

From dm1_project_guidline.md, Classification & Regression (30 pts) requires ALL of:

CLASSIFICATION (target = Rating: Low/Medium/High):
□ Decision Tree — test gain criterion (gini vs entropy) AND max_depth values
□ KNN — test multiple k values, justify best k
□ Naive Bayes — GaussianNB, discuss independence assumption
□ For each classifier: confusion matrix, accuracy, precision, recall, F1, ROC curve
□ Discuss attribute choice (why these features?)
□ Final comparison: which classifier is best and why?
□ Interpret the decision tree (what splits does it make? what does it tell us?)

REGRESSION (target = GameWeight — same target for both tasks):
□ Single regression: GameWeight ~ NumWeightVotes (1 variable, linear only)
□ Multiple regression: GameWeight ~ [2+ variables], solved with:
  □ Linear regression
  □ Non-linear approach 1 (e.g. polynomial degree 2)
  □ Non-linear approach 2 (e.g. decision tree regressor or random forest)
□ Evaluate ALL regression models: MSE and R²
□ Discuss which regression model is best and why

---

## STRICT OUTPUT RULES — NO EXCEPTIONS

1. **No fallback code.** No `try/except` that silently hides failures. No
   `if results is None: print("Using mock data")`. Every code path must
   produce real output or raise visibly.
2. **No template/placeholder markdown.** Every discussion cell must contain
   ACTUAL numbers from that cell's output. Examples of BANNED text:

   - "The Decision Tree achieved good accuracy" → BANNED (no number)
   - "KNN performed well for most classes" → BANNED (which classes? what F1?)
   - "if the tree depth is too large, it may overfit" → BANNED (generic advice)
     REQUIRED format: "Decision Tree (max_depth=5, entropy) achieved F1-macro=0.61.
     The Low class had F1=0.58, Medium=0.65, High=0.55, reflecting the class
     imbalance (Low=33%, Medium=44%, High=23%)."
3. **Narrative must match code output.** If the grid search picks max_depth=5,
   the markdown must say max_depth=5 — not "a moderate depth". If R²=0.41,
   say R²=0.41, not "moderate fit".
4. **No output-less markdown cells.** Every discussion cell must follow a
   code cell that printed the numbers being discussed. If you write a number
   in markdown, it must appear verbatim in the preceding code output.
5. **Every parameter grid must be explicit.** List ALL values tested, not just
   the winner. e.g. "We tested max_depth ∈ {3, 5, 7, 10, None} and
   criterion ∈ {gini, entropy} — 10 configs total. Best: depth=5, entropy."

---

## WINDOWS / PERFORMANCE CONSTRAINTS

- `os.environ['LOKY_MAX_CPU_COUNT'] = '4'` at top of notebook
- All sklearn calls: `n_jobs=1` (NOT -1 — causes subprocess hang on Windows)
- Cross-validation: `cv=5`, `scoring='f1_macro'`
- ROC curve: use `label_binarize` for multiclass (OvR strategy)
- StandardScaler already applied to continuous features before KNN distance

---

## DATA FACTS (use these — do not re-derive differently)

- Clean dataset: `dataset/processed/DM1_game_dataset_clean.csv`, 21925 rows × 37 cols
- Target Rating distribution: Low=7245 (33%), Medium=9644 (44%), High=5036 (23%) — imbalanced
- Features for classification: the 15 continuous clustering features
  (YearPublished, GameWeight, MinPlayers, MaxPlayers, ComAgeRec, LanguageEase,
  BestPlayers, log_NumOwned, log_NumWish, NumWeightVotes, log_MfgPlaytime,
  MfgAgeRec, NumAlternates, NumExpansions, NumImplementations)
  + 8 binary Cat:* columns = 23 features total
- Regression target: GameWeight (continuous, range ~0–5, mean ~1.98)
- Best single predictor for GameWeight: NumWeightVotes (justify with correlation)
- Multiple regression predictors: NumWeightVotes, MfgPlaytime, ComAgeRec, NumExpansions
- Rating is already dropped from feature matrix X before training

---

## NOTEBOOK STRUCTURE (cells in order)

Cell 0  [md]  # Task 3: Classification & Regression — title + section overview
Cell 1  [md]  ## 0. Setup
Cell 2  [code] imports + os.environ + load clean dataset + print shape + Rating counts
Cell 3  [md]  ## 1. Classification — Feature Selection (justify which 23 features and why)
Cell 4  [code] build X (23 features), y (Rating encoded), train_test_split 80/20
              stratified, print class counts in train and test
Cell 5  [md]  ## 2. Decision Tree
Cell 6  [code] grid search: max_depth=[3,5,7,10,None] × criterion=[gini,entropy]
              cv=5, f1_macro — print full results table sorted by score
Cell 7  [md]  **Best DT config**: state exact values from cell 6 output
Cell 8  [code] fit best DT on train, predict test, print classification_report,
              plot confusion matrix, save to report/images/dt_confusion_matrix.png
Cell 9  [code] ROC curve (OvR) for DT — plot + save dt_roc_curve.png
Cell 10 [code] plot decision tree (max_depth=3 for readability), save dt_tree_plot.png
Cell 11 [md]  **DT Discussion**: exact F1-macro, per-class F1, tree interpretation
              (what is the root split? what feature separates High from Low?)
Cell 12 [md]  ## 3. KNN
Cell 13 [code] StandardScaler fit on train, transform train+test
              grid search k=[3,5,7,11,15,21], cv=5, f1_macro — print results table
Cell 14 [md]  **Best k**: state from output
Cell 15 [code] fit best KNN, predict, classification_report, confusion matrix,
              save knn_confusion_matrix.png
Cell 16 [code] ROC curve for KNN, save knn_roc_curve.png
Cell 17 [md]  **KNN Discussion**: exact F1-macro, per-class F1, effect of k,
              why scaling is necessary, which class is hardest
Cell 18 [md]  ## 4. Naive Bayes
Cell 19 [code] GaussianNB fit+predict, classification_report, confusion matrix,
              save nb_confusion_matrix.png
Cell 20 [code] ROC curve for NB, save nb_roc_curve.png
Cell 21 [md]  **NB Discussion**: exact F1, independence assumption discussion,
              which class benefits/suffers, comparison to DT and KNN so far
Cell 22 [md]  ## 5. Classifier Comparison
Cell 23 [code] summary table: Accuracy, F1-macro, Precision-macro, Recall-macro
              for all 3 classifiers — print as DataFrame
Cell 24 [code] side-by-side ROC curves for all 3 (3 subplots), save classifier_roc_comparison.png
Cell 25 [md]  **Final classification discussion**: which is best, which is worst,
              reference exact F1 values, discuss imbalance effect, reference tree interpretation
Cell 26 [md]  ## 6. Regression — Target and Feature Selection
Cell 27 [code] print correlation of all features with GameWeight, top 5,
              justify NumWeightVotes as single predictor
Cell 28 [md]  ## 7. Single Linear Regression: GameWeight ~ NumWeightVotes
Cell 29 [code] LinearRegression fit+predict, print MSE and R²,
              scatter plot with regression line, save regression_single.png
Cell 30 [md]  **Single regression discussion**: exact MSE and R², interpret slope
Cell 31 [md]  ## 8. Multiple Regression: GameWeight ~ [4 features]
Cell 32 [code] Multiple linear regression, print MSE + R² + coefficients table
Cell 33 [code] Polynomial regression (degree=2), print MSE + R²,
              save regression_poly.png (predicted vs actual)
Cell 34 [code] Decision Tree Regressor (max_depth grid [3,5,7,10]),
              best by CV RMSE, print MSE + R²
Cell 35 [md]  ## 9. Regression Comparison
Cell 36 [code] summary table: model, MSE, R² for all 4 regression models
              predicted vs actual scatter for all 4, save regression_comparison.png
Cell 37 [md]  **Final regression discussion**: exact MSE/R² for each,
              which model fits best, why non-linear helps (or not), limitations
