# Phishing URL detection: evaluation and deployment recommendation

## Manager summary and recommendation

I would **not automatically block URLs** with this model. At the chosen score threshold of **0.95**, it recalled **34.0%** of phishing URLs in the supplied test set and **25.7%** in the separate external test set. Their ROC-AUC values were **0.906** and **0.894**, respectively. I view these ROC-AUC scores as evidence that the model retains discriminatory power on external public dataset too, that is different from the training set, however this doesn't indicate performance in production, especially with the very different phishing prevalence.

When estimating expected deployment performance, I estimated expected precision using the expected phishing prevalence of **0.1% (1 in 1,000 URLs)**. At higher thresholds, no false positives were observed in the external test set, causing the estimated deployment precision to reach 100%. This should **not** be interpreted as perfect expected production precision: the finite number of negative observations in the test set limits our ability to reliably estimate the very small false-positive rates that become critical at such a low phishing prevalence. In the test set the estimated deployment precision at the chosen threshold was 4.2%, which is close to the one measured on the training set (using cross-validation) 4.4%. The difference between these two results can come from the two test sets having different distributions and differently structured benign URLs.

I would first run the model in shadow mode to measure the performance in production because if it's precision is close to the test set precision of 4.2% this would result in a huge number of benign URLs getting flagged (768 out of 10 000) eroding human trust in the system. All the while the system probably wouldn't catch every phishing URL, only 1 in 3 or 1 in 4 as per the test results. This could be possible with a human review queue on flagged URLs, which would allow calculation of deployment precision.

## Data, observations, and model

The training file contained 8,651 rows and repeated URLs from merged feeds. For cleaning, I lowercased and stripped URL text, removed its scheme, a leading `www.`, and trailing slashes to form a *deduplication key*. This removed all 107 rows in 50 groups with conflicting labels, then 219 additional repeated rows by retaining the first row in each nonconflicting group. **8,325 rows** remained. Feature extraction still uses the raw URL.

I engineered features that describe a URL’s structure, character patterns, and potential warning signals rather than using the URL text as an identity. They include lengths and path structure, such as URL length and number of path segments; character composition, such as digit ratio and entropy; and indicators for query strings, IP-address hosts, suspicious keywords, and certain top-level domains. My initial choices came from my own reasoning, experience with phishing-looking URLs, and examples I found online; I then checked which patterns were supported by the training data.

Three patterns in the cleaned training collection informed the model:

1. A query string occurred in **24.5%** of phishing URLs versus **3.4%** of legitimate URLs, supporting parsed URL and query-structure features.
2. A suspicious-keyword flag matched **1,368/8,325** URLs; **90.1%** of those matches were labeled phishing.
3. I expected broad redirect words to be reliable signals, but case-insensitive matching of `out` found **212** URLs and only **39.6%** were phishing. Removing it from the redirect-word list reduced matches from **460** to **261** and raised the phishing share among matches from **54.8%** to **65.9%**.

To this data I fitted multiple logistic regressions, each with different feature subsets. The final model has **44 URL-derived features**, it includes all features but the ones that carry little to no signal according to the exploratory data analysis. In training-only cross-validation, its out-of-fold ROC-AUC was **0.908**, versus **0.883** for the structural baseline. Regularization was selected on training data (`C=20`), and the final pipeline was refitted on all cleaned training rows.

Four consequential decisions were:

- conflicting and repeated keys → clean before splitting → 326 rows removed and less duplicate exposure across folds
- dropping features that carried little to no signal according to the EDA; while these features improved performance on the training set, a model including them performed slightly worse on the test sets
- noisy `out` matches → drop that term → a narrower redirect flag
- keyword-based risk indicators produced the largest improvement over the structural baseline.

## Test set assessment and overlap

Both supplied test files were reserved for assessment. The table uses the **0.95 operating threshold**; precision here reflects the files' roughly 50% phishing prevalence. The regular test was reduced from 2,858 to **2,834 distinct normalized URLs** for evaluation; the external test retained all 2,858. The submitted `predictions.csv` retains **all 2,858 original test rows** in input order.

| Evaluation set | Recall | FPR | Precision | ROC-AUC | FP / legitimate URLs |
|---|---:|---:|---:|---:|---:|
| Regular test | 34.0% | 0.77% | 97.8% | 0.906 | 11 / 1,429 |
| External test | 25.7% | 0/1,429 observed | 100% observed | 0.894 | 0 / 1,429 |

Using the deduplication key as an exact-overlap definition, **138/2,834 (4.9%)** regular-test rows and **0/2,858** external-test rows matched training. The regular test still shares registered domains with training for **1,349/2,834 (47.6%)** rows, versus **330/2,858 (11.6%)** externally. The lower external recall and AUC are observed; different source composition is a plausible explanation, but there are multiple possible explanations.

## Threshold, base rate, and limits

In the absence of specified costs for false positives and false negatives, I selected **0.95 as an illustrative conservative operating point** from the OOF threshold analysis. It substantially reduces false positives relative to the default 0.50 threshold while retaining some phishing recall. I do not claim that 0.95 is an optimal production threshold. Relative to 0.50, OOF FPR fell from **14.6% to 0.78%**, while recall fell from **80.4% to 35.9%**. This is a trade-off for review prioritization, not a threshold demonstrated safe for blocking. The same 0.95 threshold is used by the prediction function and held-out assessment.

At the assumed deployment prevalence of one phishing URL per 1,000, I estimate alert precision from `TPR × prevalence / (TPR × prevalence + FPR × (1 − prevalence))`. Applying the OOF rates on the training set gives about **4.4% expected precision**: roughly **36 phishing URLs caught and 774 legitimate URLs falsely flagged per 100,000 incoming URLs**. The regular test rates imply a similar **4.2%**. This extrapolation assumes the class-conditional recall and FPR transfer to deployment; the observed source differences make that uncertain. Model probabilities were fitted on approximately balanced feeds and are **not calibrated to deployment prevalence**. Empty or malformed input strings are scored through the same feature pipeline; non-string inputs are rejected.

The external set's zero false positives do **not** establish a zero FPR or 100% deployed precision. With only 1,429 legitimate URLs, a rough 95% upper bound after zero observed errors is about **3/1,429 = 0.21% FPR**; at its measured recall, even that rate would imply only about **11%** deployment precision. Before warning users or blocking, I would need a larger, time-separated sample of real incoming URLs, verified labels, measured prevalence and alert workload, subgroup/source checks, and an agreed cost or capacity target.

## Next steps

I would deepen the EDA and review misclassified URLs to understand which patterns the model misses and which legitimate URLs it flags also a deeper domain knowledge would go a long way in improving feature engineering. That would guide more targeted features, including ideas from classical text analysis. One idea I am interested in is whether deliberate misspellings can be detected and used features somehow (e.g. paypa1-secure-login or mircosoft, etc). I would also test character n-grams with a linear model and see if that approach could generalize well outside the training set. Before deployment, I would evaluate the model on representative incoming traffic, then revisit the threshold and probability calibration for the much lower phishing prevalence.