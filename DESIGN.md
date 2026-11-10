# Design notes

This is the technical companion to the README: why each decision went the way it did, and the run
that backs it. The notebooks walk through every step with its output. Every number here is printed
by code in this repo.

![How the code is laid out](assets/architecture.png)

## Choosing the problem

I first checked which outcomes actually move with things known in advance
([notebook 02](notebooks/02_where_the_signal_is.ipynb), `scripts/explore.py`).

| Candidate | Overall | Range across the obvious drivers | Verdict |
|---|---|---|---|
| Invoice never paid | 23.8% | 21 to 26% by customer type, source, industry, amount, payment terms | nothing to learn |
| Visit cancelled | 9.7% | 7 to 11% by type of work and industry; reschedules matter but happen after booking | too late to act on |
| Business leaves Jobber | 30 of 500 | too few to learn from or test | too rare |
| Quote accepted | 67.3% | 55 to 79% by type of work, 58 to 76% by customer source, 61 to 73% by sending speed | **chosen** |

Unpaid invoices surprised me, since getting paid is the obvious pain point in the brief. But a model
there would only be fitting noise. Businesses also don't differ beyond their mix of work and
customers (scatter 1.10 times what chance predicts, p = 0.06), so one shared model serves all 500.

## Reading the data

`quote-followup audit --zip ../data.zip` runs 86 checks, and all pass
([notebook 01](notebooks/01_reading_the_data.ipynb)). The files match the zip byte for byte, row
counts match a plain line-by-line read, every value converts to the type the data dictionary gives,
required fields are never empty and ids are unique. Every link in the entity diagram points to a real
record in the same business, and my daily counts of quotes created, sent and accepted match
`daily_sp_metrics` exactly. The loader converts dates using the files' exact format and stops on
anything else. The usual pandas shortcut would leave a bad date as text without a word of warning.

Four facts from the data shaped the design:

- The extract was taken on or after 19 Mar 2025, and every quote had expired by 23 Jan. So the
  1,444 quotes still marked "open" are stale, not waiting.
- Only 6 of 19,493 replies came more than 45 days after sending.
- A won job's planned and invoiced revenue are within 2.5% of the quote for every type of work, so
  the quote amount is a fair value.
- Lost quotes end differently depending on the customer (next section but one).

## Never seeing the future

Every fit and every score takes a date, and `timeline.py` is the only way to get data for it.
`open_quotes(as_of)` removes every column that records how a quote ended, so the scoring code
can't read them. `settled_quotes(as_of)` returns quotes sent at least 45 days earlier whose ending was
on record by that date. The 45-day wait matters: among recent quotes the quick yeses are in but the
slow nos aren't, so learning from them would make yeses look more common than they are.
`tests/test_point_in_time.py` rewrites every outcome after the date and checks the scores don't move.

## The model

A quote ends in a yes, a no, or silence, and the model is built around that:

- A logistic regression on the three facts gives *p*, the chance of a yes when the quote goes out.
- A second one, fitted on lost quotes, gives *q*, the chance a no is said out loud rather than by going quiet.
- `DecisionTiming` learns, per type of work, how many yeses and nos are still to come after *a* days.

Combined: `logit P(yes | no reply after a days) = logit p + log S_yes(a) − log[q·S_no(a) + (1 − q)]`.
This is the mixture model for competing outcomes of Larson and Dinse (1985); the cure model (Farewell
1982) is its two-outcome case. It assumes reply speed depends on the type of work and the answer,
nothing else. Notebook 03 shows that holds well for recurring and emergency work and less well for
one-off projects. The reply curves count one pretend quote that never replies, so they start at
exactly 1 and never reach 0.

*q* is there because repeat and referred customers who say no mostly say it, while marketplace and
self-found leads mostly go quiet (chi-square p < 10⁻¹¹ within every type of work). With one loss curve
per type of work, a quiet repeat customer looked like a marketplace lead gone cold, and the first
version underrated those quotes by up to 11 points.

## Choosing how to build it

Every choice was made on practice runs from April to July, with replies settled by 1 September, so
the final test months were never looked at ([notebook 04](notebooks/04_choosing_the_method.ipynb)).
The yardstick is log loss; ties go to whichever holds up better group by group.

| Which facts (5,990 quotes) | AUC | Log loss |
|---|---|---|
| **type of work, customer source, sending speed** | **0.639** | **0.614** |
| + industry, customer type, quote size, number of items | 0.637 | 0.615 |
| + the business's / the customer's past win rate | 0.639 / 0.636 | 0.614 / 0.616 |
| without type of work / customer source / sending speed | 0.581 / 0.623 / 0.630 | 0.632 / 0.619 / 0.617 |
| gradient boosting with every fact | 0.587 | 0.709 |

| Using days without a reply (24,151 quote-days) | Log loss | Groups missed beyond luck |
|---|---|---|
| **says-no-or-goes-quiet chance per customer** | **0.5705** | **1** |
| one figure per type of work (first version) | 0.5717 | 3 |
| reply speed also by sending speed / by customer source | 0.5704 / 0.5711 | 3 / 2 |
| days waiting as one more fact (van Houwelingen 2007) | 0.5733 | 4 |
| gradient boosting, either way | 0.614 / 0.626 | 7 / 3 |
| days waiting only | 0.5920 | 27 |

## Testing it

The four claims, how each is tested and what would count against it are in
[notebook 03](notebooks/03_hypotheses.ipynb). The results are in
[notebook 05](notebooks/05_testing_the_hypotheses.ipynb). Three choices keep the test honest:

- **Replay forward, relearning monthly**, only from replies already in (Tashman 2000).
- **Count total dollars, not an average of mornings.** An owner cares about the total. An average of
  each morning's share counts a $60 morning the same as a $6,000 one; on it, our list trails newest
  first (0.665 against 0.710). I switched measures after the first run, and both are reported.
- **Re-draw whole businesses, the same way for every method.** One business's mornings are alike
  (Field and Welsh 2007), and identical re-draws make the comparisons like-for-like. The headline
  ranges are also shown widened for comparing six methods at once (Bonferroni).

The group-by-group check gives each group a z-score, treating a quote's many mornings as one unit.
My first version used "the biggest gap in any group", which is a maximum of noisy numbers and always
looks bad. The pooled AUC of 0.87 has a related problem: quotes ignored for weeks count once per
morning as easy nos. The fair figures are each quote counted once (0.69) and quotes of the same age
(0.66 to 0.69).

## What the second review changed

The first version had a three-fact model, one reply-speed curve per type of work, and a README
saying it beat the rules of thumb by 17 points. A second look found:

1. **Model choices had been made looking at the test months.** I re-made them on April to July; they
   stood.
2. **Being right on average hid being wrong for some groups.** The average gap was 0.006, but
   recurring quotes from referred customers, 3 to 7 days old, were underrated by 11 points. Splitting
   reply speed by group or adding interactions didn't fix it; the says-no-or-goes-quiet chance did.
3. **The toughest simple rival was missing.** Adding "days waiting × amount" showed the order owes
   nothing measurable to the facts about the quote. The README says so plainly.
4. **The loader could fail without anyone noticing**, and nothing proved the data was read as
   supplied. The strict loader and the 86-check audit fix both.
5. **Smaller things.** The chart title "slow quotes lose" implied a cause. The reply curves gave a
   small false signal on day 0. "The dollars a call is chasing" implied a call causes the yes.
6. **Wording.** Every label, chart, comment and notebook was rewritten for readers outside data science.

## How the code is laid out

| File | Its one job | Changes when |
|---|---|---|
| `data.py` | read the files strictly and join them within each business | the data source changes |
| `audit.py` | prove the files are read as supplied | the data's rules change |
| `timeline.py` | what was known on a given date | the meaning of "known" changes |
| `propensity.py` | the explainable scoring model | the facts used change |
| `timing.py` | reply speeds and what silence means | the silence model changes |
| `scorers.py` | our list, the simple rivals, and the shared interface | a new way of ordering arrives |
| `board.py` | the list an owner reads | the output changes |
| `backtest.py` | the replay and its checks | the test changes |

The principles, and where each shows up:

- **One job per file**, as in the table. Charts and notebooks live outside the package, so the core
  needs only numpy, pandas, scipy and scikit-learn.
- **New ideas plug in without edits.** A new ordering (an uplift model, say) implements `fit` and
  `priority` and joins the replay. The facts and reply-speed groups are lists in `config.toml`.
- **Everything is swappable through one small interface.** The replay calls our model and every rival
  the same way and can't tell them apart, which is what makes the comparison fair. It depends on that
  interface, not on any one model.
- **Rules are checked where data comes in.** The loader stops on a customer filed under the wrong
  business, an unknown status or a badly written date.
- **Reproducible by construction.** Every setting is in `config.toml`, the re-draws are seeded, the
  "random" order is a hash of the quote id, the notebooks are built from plain sources by
  `notebooks/build.py`, and `make all` rebuilds everything from the CSV files.

## References

- Farewell, V. T. (1982). The use of mixture models for the analysis of survival data with long-term survivors. *Biometrics* 38(4).
- Field, C. A. and Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society B* 69(3).
- Larson, M. G. and Dinse, G. E. (1985). A mixture model for the regression analysis of competing risks data. *Applied Statistics* 34(3).
- Naeini, M. P., Cooper, G. and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI*.
- Radcliffe, N. J. and Surry, P. D. (2011). Real-world uplift modelling with significance-based uplift trees. Stochastic Solutions white paper.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting* 16(4).
- van Houwelingen, H. C. (2007). Dynamic prediction by landmarking in event history analysis. *Scandinavian Journal of Statistics* 34(1).
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22(158).
