# Red-team upload7 no_month — 2026-06-19

| check | evidence | risk | verdict |
|---|---:|---|---|
| Offline strength | Fold3 `0.7584`; OOF `0.7822`; lh_mean `0.7681`; lh_min `0.7583` | local validation may not transfer | PASS_WITH_RISKS |
| Versus upload3 | lh_mean `+0.0016`; lh_min `+0.0004`; Fold3 `+0.0003` | gain is modest but consistent | PASS_WITH_RISKS |
| Ablation overfit | `no_month` selected after ablation scan; month feature removal helps late holdouts | ablation-selected branch can overfit diagnostics | RETEST |
| Time extrapolation | keeps day-num/week, removes month categorical | still calendar-sensitive | RETEST |
| Format/hash | rows/order/range OK; SHA256 `4e26ffecbb1bdc169c1595ba2a950550a08c594218997501f4915098c88c412e` | none found | PASS |
| Upload folder | only `upload7` and `upload1` CSV remain | no duplicate-slot risk | PASS |

Verdict: `upload7_no_month_c20_n80` is the current best risk-taking candidate. Keep `upload1_c35_x65` as fallback. Do not upload archived `upload3/upload5/upload6`.

Validation level: L4 diagnostics + L5 format/hash check; leaderboard transfer is not proven.
