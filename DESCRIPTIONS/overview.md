Overview
BlueShell Insurance Company receives thousands of auto insurance quote requests daily through multiple channels (online, agent, call center, aggregator). Only a fraction of these quotes convert into bound policies. This dataset supports the development of a quote conversion prediction model — identifying which prospects are most likely to purchase — to improve follow-up prioritization, marketing efficiency, and pricing strategy.

Conversion is a critical concept in insurance sales. Every day, thousands of prospective customers request a quote — online, through an aggregator/comparison site, over the phone, or through an agent — but only a fraction turn into a purchased ("bound") policy. Understanding which quotes are likely to convert, and why others are lost, allows BlueShell to focus sales follow-up on the right prospects, price more competitively where it matters, and avoid wasting underwriting and marketing resources on quotes that were never going to bind.

For example, a prospect gets an online quote and spends several minutes revising their coverage before abandoning the page. A call center agent later follows up and learns in the call notes that the customer thought the price was too high compared to their current carrier. BlueShell needs to know, ideally before that lost conversion happens, that this quote was at risk — so it can intervene with a better offer, a discount, or a well-timed callback.

Goals of this competition
Identify the likelihood of conversion (bind) for incoming auto insurance quotes.

Understand key indicators of quote conversion, including pricing competitiveness, channel, and consumer behavior signals.

Provide a recommendation on how this information could be leveraged at BlueShell Insurance Company to improve conversion rates and marketing/sales efficiency.

Keywords
Quote – a premium estimate generated for a prospective customer before a policy is purchased.

Bind / Conversion – when a quoted prospect completes the purchase and activates a policy.

Channel – the path through which a quote was generated (online direct, call center, agent, or aggregator/comparison site).

Shopping behavior – the pattern of a consumer comparing multiple quotes or carriers before making a purchase decision.

Aggregator – a third-party comparison site (e.g., a "shop around" platform) where a consumer receives quotes from multiple carriers at once.

Modeling
Each group may have at most 5 people and will: a. Work together within groups but not between groups (They can’t provide extra info or help. Please do not use the Discussion board – contact us directly if anything comes up). b. Build a model to identify the likelihood of conversion (bind) for incoming auto insurance quotes. c. Submit the prediction of the test data as a csv file. (The format requested is provided in the attachment.) d. Prepare a presentation for your business partner to summarize your analysis results. You do not need to explain the problem, just summarize what you did and what you found (see questions to answer in section 'Presentation Instructions' below). e. Each group can make at most 3 submissions per day.

Benchmark Model
A gradient boosting machine (GBM) model serves as the competition benchmark.

Disclaimer
BlueShell Insurance Company and the data is a fictitious example used for the purpose of this competition only.

Model Evaluation & Competition Logistics
Model Evaluation Metric
Submissions are scored against the withheld validation labels using the metrics below. Because the model outputs conversion probabilities (not hard classifications), all metrics are threshold-free unless noted.

Primary ranking metric: AUC-ROC. Tie-breaking will use Precision at the top decile (higher is better), which is fraction of policies in the top-decile score band that are true converters (true positives ÷ total flagged at that cutoff).

Teams should report both metrics on the holdout set in their presentation, along with a decile-level lift chart and calibration plot showing predicted vs. observed conversion rates by score band.