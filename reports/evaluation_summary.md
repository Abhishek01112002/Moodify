# Evaluation Summary

Command:

```bash
python scripts/evaluate_recommender.py --data spotify_songs_cleaned.csv --top-k 10 --sample-size 200 --output reports/evaluation_sample.json
```

Dataset:

- Rows evaluated: 525
- Sampled seed tracks: 200
- Proxy relevance label: `mood_label`
- Feature count: 22

Results:

| Metric | Value |
| --- | ---: |
| Precision@10 | 0.472 |
| Recall@10 | 0.047 |
| MAP@10 | 0.364 |
| NDCG@10 | 0.491 |
| Catalog coverage | 0.918 |
| Artist diversity | 0.203 |
| Avg latency/query | 2.08 ms |

Interpretation:

The recommender is fast and covers most of the catalog during sampled
evaluation. Precision@10 and NDCG@10 show that many top recommendations share
the seed track's mood label. Artist diversity is the clearest next improvement:
the ranking should reduce repeated artists and expose more variety.

Important caveat:

These are proxy-label offline metrics. For production-quality evaluation, use
real user behavior such as likes, saves, playlist additions, skips, repeat
plays, and search-to-play events.
