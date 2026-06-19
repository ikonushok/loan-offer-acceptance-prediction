# Red-team upload3_time_regime — 2026-06-19

| check | evidence | risk | verdict |
|---|---:|---|---|
| Offline strength | Fold3 `0.7571`; blend lh_mean `0.7665`; lh_min `0.7579` | may not transfer to public/private | PASS_WITH_RISKS |
| Versus upload1 | upload1 lh_mean `0.7641`; lh_min `0.7563` | gain is modest | PASS_WITH_RISKS |
| Time-regime extrapolation | uses time-regime signal; strongest on late holdouts | calendar/test drift overfit possible | RETEST |
| Three-way blend | best weights collapse to `c20/x00/t80` | XGB HPO adds no diversity | HOLD |
| Format/hash | rows/order/range OK; SHA256 `46d7f44306626b6a2b6b65bb2aaa4defca61165b58988c081de16a770fc8668b` | none found | PASS |
| Duplicate candidate | upload4 has same SHA as upload3 | wastes upload slot / confusion | BLOCK duplicate |

Verdict: `upload3_time_regime_c20_x80` is the best risk-taking upload candidate; keep `upload1_c35_x65` as conservative fallback. Do not upload `upload4`.

Validation level: L4 diagnostic + L5 format/hash check; leaderboard transfer is not proven.
