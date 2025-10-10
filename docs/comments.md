
Major issues that need to be resolved in the revision:
1. Inconsistent reported numbers — unify and verify all values
Several key statistics appear with different values in different places (for example, reported Cohen’s κ and correlation coefficients appear as different numbers in the response letter, abstract, body, and conclusion). The authors must verify and unify all reported numbers across the manuscript and response letter. If different computation methods were used, explicitly state and compare them. Provide the raw summary tables or scripts used to compute each aggregated number.

2. Reproducibility and methodological detail are insufficient
(1)Embedding model and hyperparameters: Specify exactly which embedding model(s) were used (model name and checkpoint), tokenizer settings, any fine-tuning, and preprocessing steps (lowercasing, stopword removal, lemmatization, frequency thresholds).
(2)LLM call details: Report the exact LLM model/version used, date of calls, API parameters (temperature, top_p, max_tokens), number of evaluations per item, aggregation method (mean/median), and whether sampling was deterministic. Include pseudo-code or actual scripts showing how continuous LLM scores were converted to categorical labels and then to Cohen’s κ.
(3)Dataset construction & availability: For any datasets built (Wikipedia-derived or otherwise), provide the crawl date, query seeds or page lists, filtering rules, and several example documents per topic. Ideally release the dataset or provide code to reproduce it.

3. Metric definitions and normalization are unclear
Some custom metrics (e.g., Semantic Coherence, Semantic Distinctiveness, SemDiv with α/β/γ/λ parameters) lack full mathematical specification, parameter values, and value ranges. Provide precise formulas, show chosen parameter values with justification, and include a small worked example (toy data) to demonstrate calculation end-to-end.

4. Discussion of LLM evaluation limitations & robustness tests
Acknowledge bias and hallucination risks of LLMs and test robustness: run sensitivity analyses across different temperature settings, prompt variants, and ideally across more than one LLM. Present how much scores vary and discuss mitigation strategies (e.g., multi-model consensus, prompt ensembling).

Minor issues that recommended fixes:

(1)Table and figure clarity
Improve table layout (clear headers, consistent column alignment) and add a concise caption and a one-sentence reader takeaway for each table.
For t-SNE plots, add hyperparameters (perplexity, learning rate, seed) and consider adding a UMAP comparison and/or multiple seeds to demonstrate stability.

(2) Terminology and abbreviation consistency
Ensure every abbreviation (e.g., NPMI, IRBO) is defined upon first use. Run a global search to catch missed items.

(3) Appendix code and pseudo-code
The appendix pseudocode is useful — complement it with minimal runnable examples for key routines: computing semantic metrics, LLM scoring, and Cohen’s κ aggregation.

(4) Language polish
The manuscript reads well overall but contains occasional redundancy and formatting issues. A final round of native-English proofreading (or professional editing) is recommended.

(5) Conclusion alignment
Make sure the conclusion’s numeric claims and limitation statements match the body and abstract. Explicitly list main limitations and concrete future work items (multi-lingual extension, low-resource behavior, reducing LLM cost).




1. Please add at least one simple public real-world dataset, because relying solely on three Wikipedia-based synthetic datasets limits external validity.

2. Clarify Related Work regarding Ref. 15. You cite LLM-based evaluation in Related Work, but §2.2 frames the “existing metrics” as statistical. Please simply state how your metric differs and why it is more important.

3. Specify metric details in §3.3. State exactly which neural embedding model you use (only the 384-dimensional setting is given), how λw is chosen or learned, and what values or selection process you used for α, β, γ.

4. Fix numeric inconsistencies. For example, the Conclusion reports κ = 0.89, which differs from other sections.

