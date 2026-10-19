# test split

50 tasks. `recall@k`: share of a task's required documents in the top k. `tools@k`: share of the discoverable tools in the reference trajectory that some top-k document names. `hit@k`: at least one required document in the top k.

| method | query | hit@5 | recall@5 | recall@10 | recall@20 | tools@10 | tools@20 | MRR | ms/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tau2 bm25 (baseline) | opening | 34.0% | 4.3% | 6.3% | 13.6% | 4.7% | 17.6% | 0.258 | 9.3 |
| bm25 (ours) | opening | 66.0% | 14.5% | 23.8% | 32.1% | 27.8% | 33.5% | 0.461 | 6.9 |
| dense (LSA) | opening | 66.0% | 13.7% | 23.3% | 37.7% | 30.5% | 43.2% | 0.466 | 18.8 |
| hybrid | opening | 74.0% | 17.6% | 24.6% | 40.4% | 31.0% | 44.1% | 0.529 | 23.4 |
| tau2 bm25 (baseline) | scenario | 50.0% | 8.8% | 17.0% | 22.4% | 27.5% | 38.6% | 0.361 | 80.7 |
| bm25 (ours) | scenario | 86.0% | 26.9% | 34.5% | 43.5% | 53.2% | 64.4% | 0.721 | 17.5 |
| dense (LSA) | scenario | 76.0% | 23.9% | 35.9% | 49.2% | 52.9% | 73.0% | 0.573 | 22.3 |
| hybrid | scenario | 86.0% | 27.2% | 37.0% | 46.8% | 56.7% | 70.3% | 0.727 | 39.8 |
