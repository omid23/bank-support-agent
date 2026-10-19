# dev split

47 tasks. `recall@k`: share of a task's required documents in the top k. `tools@k`: share of the discoverable tools in the reference trajectory that some top-k document names. `hit@k`: at least one required document in the top k.

| method | query | hit@5 | recall@5 | recall@10 | recall@20 | tools@10 | tools@20 | MRR | ms/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tau2 bm25 (baseline) | opening | 14.9% | 2.8% | 5.8% | 11.9% | 7.2% | 14.0% | 0.101 | 10.1 |
| bm25 (ours) | opening | 57.4% | 13.9% | 21.6% | 34.4% | 22.2% | 38.3% | 0.321 | 5.9 |
| dense (LSA) | opening | 61.7% | 15.2% | 23.1% | 39.6% | 24.9% | 46.3% | 0.458 | 19.2 |
| hybrid | opening | 70.2% | 18.0% | 25.5% | 39.8% | 26.8% | 44.3% | 0.478 | 24.7 |
| tau2 bm25 (baseline) | scenario | 44.7% | 7.4% | 12.5% | 17.7% | 26.2% | 40.4% | 0.327 | 85.2 |
| bm25 (ours) | scenario | 80.9% | 21.1% | 28.3% | 41.5% | 46.4% | 68.5% | 0.689 | 17.8 |
| dense (LSA) | scenario | 80.9% | 21.7% | 31.0% | 41.9% | 45.3% | 70.5% | 0.617 | 19.1 |
| hybrid | scenario | 80.9% | 21.7% | 30.1% | 42.2% | 52.8% | 68.3% | 0.684 | 39.1 |
