# Response to Reviewer Comment on Dataset Selection

## Reviewer Comment

> "Please add at least one simple public real-world dataset, because relying solely on three Wikipedia-based synthetic datasets limits external validity."

---

## Our Response

We sincerely appreciate the reviewer's concern regarding external validity and the suggestion to include additional datasets. We would like to respectfully clarify our dataset design and provide justification for our methodological choices.

### Clarification on "Synthetic" Terminology

We acknowledge that our use of the term "synthetic datasets" may have created confusion. **We would like to emphasize that our datasets are not synthetic in the traditional sense** (i.e., artificially generated or simulated text). Rather, they consist entirely of **authentic Wikipedia articles written by human contributors**.

The term "synthetic" in our manuscript was intended to indicate that we **systematically curated** these real-world documents to create controlled experimental conditions, not that the text itself was artificially generated.

**To address this confusion, we will revise our terminology throughout the manuscript**:
- "Synthetic datasets" → **"Controlled real-world datasets curated from Wikipedia"**
- We will explicitly state in §3.1 that all documents are authentic Wikipedia articles

### Justification for Our Dataset Design

Our dataset construction approach was carefully designed to address our specific research objectives. We respectfully submit the following justifications:

#### 1. **Controlled Experimental Design is Essential**

The primary objective of our research is to **systematically evaluate the discriminative power** of semantic-based metrics compared to traditional statistical metrics. This requires:

- **Graduated topic similarity levels**: Three distinct tiers (inter-topic similarity: 0.21 → 0.48 → 0.67)
- **Controlled variables**: Consistent document quality, length, and style across similarity levels
- **Systematic comparison**: Ability to isolate the effect of topic overlap on metric performance

**Existing benchmark datasets** (e.g., 20 Newsgroups, Reuters-21578) have **fixed, uncontrolled topic overlap patterns** that would not allow us to demonstrate how metrics perform across varying degrees of semantic similarity—the core contribution of our work.

#### 2. **Wikipedia Provides High-Quality Real-World Text**

We selected Wikipedia as our data source because it offers:

- **Authenticity**: Peer-reviewed, high-quality articles written by human contributors for real-world use
- **Diversity**: Comprehensive coverage across scientific, technical, and general knowledge domains
- **Consistency**: Relatively uniform writing style and quality standards, enabling controlled comparison
- **Reproducibility**: Publicly accessible content allowing full experimental replication

#### 3. **Our Approach Balances Internal and External Validity**

**Internal Validity** (prioritized for our research question):
- ✓ Controlled manipulation of independent variable (topic similarity)
- ✓ Systematic isolation of confounding factors
- ✓ Reproducible experimental conditions

**External Validity** (addressed through dataset characteristics):
- ✓ Real-world text (authentic Wikipedia articles)
- ✓ Diverse domains (evolution, mechanics, AI, robotics, data analytics, etc.)
- ✓ Natural language written for genuine communicative purposes

### Limitations and Future Work

We acknowledge the reviewer's valid concern about generalizability to other text genres. **We will add the following to our revised manuscript**:

**In §6 (Conclusion - Limitations)**:
> "While our datasets comprise authentic Wikipedia articles covering diverse domains, the exclusive use of encyclopedia-style text represents a potential limitation. Future research should validate our semantic metrics on varied text genres (news articles, academic papers, social media posts, domain-specific corpora) while maintaining controlled similarity structures to ensure broader generalizability across different writing styles and discourse types."

**In §5 (Discussion)**:
> "Our controlled curation approach using Wikipedia articles provides the necessary experimental control for systematic metric evaluation. However, we recognize that Wikipedia's encyclopedia style may differ from other text types (e.g., conversational social media text, highly technical academic writing, or informal blog posts). We encourage future research to extend our framework to these diverse genres while preserving the graduated similarity structure essential for evaluating discriminative power."

### Optional: Supplementary Validation (If Required)

If the reviewer considers it essential, we are willing to add a **supplementary validation experiment** using an existing benchmark dataset (e.g., 20 Newsgroups). However, we would need to clearly note that:

1. Such datasets have **fixed topic overlap patterns** that do not allow graduated similarity evaluation
2. This would serve only to demonstrate **general applicability**, not to validate our core finding about discriminative power across similarity levels
3. The results would be presented as **supplementary material** to avoid diluting our main experimental narrative

We believe this addition is not necessary given that:
- Our datasets already use real-world text
- Our research question requires controlled conditions
- The limitation is already acknowledged

However, **we defer to the reviewer's judgment** on whether this supplementary validation would strengthen the manuscript.

### Proposed Manuscript Revisions

**1. Abstract**
```
Current: "...experiments with three synthetic datasets..."
Revised: "...experiments with three controlled real-world datasets
          curated from Wikipedia, representing varying degrees of
          topic overlap (inter-topic similarity: 0.21, 0.48, 0.67)..."
```

**2. Section 3.1 - Add Justification Paragraph**
```
We selected Wikipedia as our data source for three key reasons:

(1) Authenticity: Wikipedia provides high-quality, peer-reviewed
    real-world text written by human contributors across diverse domains

(2) Experimental Control: The broad topical coverage enables systematic
    selection of topic combinations with graduated similarity levels,
    essential for evaluating discriminative power of metrics

(3) Reproducibility: Wikipedia's public accessibility ensures full
    experimental reproducibility

While existing benchmark datasets (e.g., 20 Newsgroups, Reuters-21578)
are widely used in topic modeling research, they possess fixed topic
overlap patterns that would not allow evaluation of metric performance
across our three-tier similarity structure (0.21 → 0.48 → 0.67). Our
controlled curation approach directly addresses this methodological
requirement while using authentic real-world documents.
```

**3. Section 6 - Expand Limitations**
```
Current limitations section + add:

"Although our datasets comprise authentic Wikipedia articles, the
exclusive use of encyclopedia-style text represents a limitation
for generalizability. Future research should validate our semantic
metrics on diverse text genres (news, academic papers, social media)
while maintaining controlled similarity structures. Additionally,
multilingual validation and domain-specific adaptations (medical,
legal corpora) would further establish the framework's utility across
varied contexts."
```

### Summary

We respectfully maintain that our current dataset design is methodologically sound and appropriate for our research objectives. The datasets:

- ✓ **Use real-world text** (authentic Wikipedia articles)
- ✓ **Cover diverse domains** (15 topics spanning science, technology, AI)
- ✓ **Enable controlled experiments** (graduated similarity: 0.21 → 0.48 → 0.67)
- ✓ **Are fully reproducible** (publicly accessible sources)

We will enhance the manuscript by:

1. **Clarifying terminology** ("synthetic" → "controlled real-world curated")
2. **Adding explicit justification** for Wikipedia selection and curation approach
3. **Acknowledging limitations** regarding text genre diversity
4. **Proposing future work** for cross-genre validation

We believe these revisions address the reviewer's concerns regarding external validity while preserving the methodological rigor essential to our core contribution.

---

**We are grateful for the reviewer's careful consideration of our work and welcome any further suggestions for improvement.**

---

## Alternative Concise Version (for Response Letter)

### Comment 1: Dataset Selection

**Reviewer**: *"Please add at least one simple public real-world dataset, because relying solely on three Wikipedia-based synthetic datasets limits external validity."*

**Response**: We appreciate this concern and would like to clarify our dataset design. Our datasets are **not synthetic** in the traditional sense—they consist entirely of **authentic Wikipedia articles written by humans**. We used the term "synthetic" to indicate systematic curation, not artificial generation.

We selected Wikipedia because it provides: (1) high-quality real-world text, (2) controlled topic selection enabling our graduated similarity design (0.21 → 0.48 → 0.67), and (3) reproducibility. Existing benchmarks (20 Newsgroups, Reuters) have fixed overlap patterns unsuitable for evaluating discriminative power across similarity levels—our core contribution.

**Revisions**:
- Changed terminology: "synthetic" → "controlled real-world datasets curated from Wikipedia"
- Added justification paragraph in §3.1 explaining dataset rationale
- Expanded limitations in §6 acknowledging encyclopedia-style text and proposing future cross-genre validation
- Explicitly stated throughout that all documents are authentic Wikipedia articles

We believe these revisions address the external validity concern while maintaining the experimental control essential to our research question. If the reviewer deems it necessary, we can add supplementary validation with an existing benchmark as supporting evidence.
