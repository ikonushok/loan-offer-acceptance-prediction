# Validation drift audit

## Verdict

PASS_WITH_RISKS for using rolling time validation as the primary internal score; RETEST before model escalation.

## Context-group definition

- Context signature excludes `front_id`, `target_value`, `offered_rate`, `overdraft_limit_min`, and `overdraft_limit_max`.
- context column count: 23

## Current rolling time folds: context overlap

|   fold | cutoff     | next_cutoff   |   train_rows |   valid_rows | train_start   | train_end   | valid_start   | valid_end   |   context_overlap_groups |   train_overlap_rows |   valid_overlap_rows |
|-------:|:-----------|:--------------|-------------:|-------------:|:--------------|:------------|:--------------|:------------|-------------------------:|---------------------:|---------------------:|
|      1 | 2025-01-01 | 2025-03-01    |       123855 |         7612 | 2024-02-01    | 2024-12-30  | 2025-01-02    | 2025-02-28  |                        0 |                    0 |                    0 |
|      2 | 2025-03-01 | 2025-04-01    |       131467 |         5053 | 2024-02-01    | 2025-02-28  | 2025-03-01    | 2025-03-31  |                        0 |                    0 |                    0 |
|      3 | 2025-04-01 |               |       136520 |         8721 | 2024-02-01    | 2025-03-31  | 2025-04-01    | 2025-06-05  |                        0 |                    0 |                    0 |

## Repeated context by train month

| month   |   repeated_context_groups |   rows_in_repeated_context_groups |   max_context_group_size |
|:--------|--------------------------:|----------------------------------:|-------------------------:|
| 2024-02 |                       282 |                               835 |                       14 |
| 2024-03 |                       366 |                              1096 |                       13 |
| 2024-04 |                       407 |                              1323 |                       19 |
| 2024-05 |                       391 |                              1208 |                       13 |
| 2024-06 |                       234 |                               689 |                       11 |
| 2024-07 |                       260 |                               750 |                       11 |
| 2024-08 |                       273 |                               713 |                       10 |
| 2024-09 |                       292 |                               741 |                       10 |
| 2024-10 |                       169 |                               388 |                        6 |
| 2024-11 |                        65 |                               139 |                        3 |
| 2024-12 |                        62 |                               129 |                        5 |
| 2025-01 |                        57 |                               118 |                        3 |
| 2025-02 |                        71 |                               150 |                        3 |
| 2025-03 |                        93 |                               200 |                        6 |
| 2025-04 |                        74 |                               158 |                        4 |
| 2025-05 |                        78 |                               163 |                        4 |
| 2025-06 |                         7 |                                15 |                        3 |

## Conflicting target repeated context by train month

| month   |   conflicting_groups |   rows_in_conflicting_groups |   positives |
|:--------|---------------------:|-----------------------------:|------------:|
| 2024-02 |                   10 |                           35 |          10 |
| 2024-03 |                    7 |                           22 |           7 |
| 2024-04 |                    7 |                           44 |           7 |
| 2024-05 |                    6 |                           22 |           6 |
| 2024-06 |                   10 |                           39 |          10 |
| 2024-07 |                   14 |                           50 |          14 |
| 2024-08 |                   11 |                           35 |          11 |
| 2024-09 |                   19 |                           65 |          19 |
| 2024-10 |                   13 |                           34 |          13 |
| 2024-11 |                    5 |                           11 |           5 |
| 2024-12 |                    7 |                           14 |           7 |
| 2025-01 |                    4 |                            8 |           4 |
| 2025-02 |                    8 |                           16 |           8 |
| 2025-03 |                    8 |                           16 |           8 |
| 2025-04 |                    9 |                           18 |           9 |
| 2025-05 |                    8 |                           17 |           8 |
| 2025-06 |                    1 |                            3 |           1 |

## Top univariate train-vs-test drift

| column                            | dtype   |   adv_auc_value |   adv_auc_missing |   train_missing |   test_missing |   train_nunique |   test_nunique |
|:----------------------------------|:--------|----------------:|------------------:|----------------:|---------------:|----------------:|---------------:|
| decision_day                      | object  |        1        |        nan        |        0        |       0        |             485 |            177 |
| overdraft_limit_max               | float64 |        0.659653 |        nan        |        0        |       0        |            2056 |           1976 |
| overdraft_limit_min               | float64 |        0.636379 |        nan        |        0        |       0        |            2052 |           1950 |
| cb_rate                           | float64 |        0.630719 |        nan        |        0        |       0        |               4 |              5 |
| cnt_deb_ul_ip_30                  | float64 |        0.589742 |          0.551368 |        0.229777 |       0.127041 |            1387 |            583 |
| app_term_mean_360                 | float64 |        0.577549 |          0.556072 |        0.384761 |       0.272617 |            1780 |           1299 |
| offered_rate                      | float64 |        0.575524 |        nan        |        0        |       0        |             134 |            161 |
| cnt_deb_ul_ip_90                  | float64 |        0.573604 |          0.548669 |        0.208653 |       0.111316 |            2659 |           1251 |
| cnt_deb_loan_90                   | float64 |        0.567976 |          0.552101 |        0.216592 |       0.11239  |             897 |            738 |
| days_from_authperson_registration | float64 |        0.552526 |          0.555171 |        0.540295 |       0.429952 |            7091 |           5970 |
| count_all_corp_dashboard_events   | float64 |        0.548153 |          0.571538 |        0.352435 |       0.209358 |           17029 |          10814 |
| sum_deb_ul_90                     | float64 |        0.547226 |          0.567804 |        0.37256  |       0.236953 |           75712 |          23510 |
| fl_hdb_bki_total_active_products  | float64 |        0.545405 |          0.529659 |        0.167776 |       0.227094 |              91 |             66 |
| db_group_last                     | object  |        0.543294 |        nan        |        0.384761 |       0.272617 |               9 |              9 |
| cnt_cred_loan_90                  | float64 |        0.53997  |          0.552101 |        0.216592 |       0.11239  |             137 |            118 |
| sum_deb_ul_30                     | float64 |        0.539806 |          0.568279 |        0.423111 |       0.286552 |           68585 |          21711 |
| corp_credit_products              | float64 |        0.53098  |          0.571538 |        0.352435 |       0.209358 |            2504 |           1808 |
| loan_amount_last                  | float64 |        0.525418 |        nan        |        0        |       0        |            3205 |           1233 |
| overdraft_app_term_max_360        | float64 |        0.521614 |          0.520982 |        0.96207  |       0.920107 |               6 |              5 |
| loan_rev_max_start_non_fin        | float64 |        0.511768 |          0.528014 |        0.913206 |       0.857178 |             723 |            698 |
| sum_deb_investment_90             | float64 |        0.51124  |          0.555891 |        0.886093 |       0.774311 |            8769 |           5377 |
| fl_adminarea                      | object  |        0.511028 |        nan        |        0.298194 |       0.317094 |              83 |             80 |
| loan_rev_min_start_fin            | float64 |        0.509552 |          0.558676 |        0.858614 |       0.741263 |            1822 |           1578 |
| balance_rur_amt_30_min            | float64 |        0.50802  |          0.543577 |        0.239781 |       0.152626 |           71423 |          19854 |
| p75_time_spent_minutes            | float64 |        0.507653 |          0.571538 |        0.352435 |       0.209358 |           34323 |          17015 |
| corp_list                         | float64 |        0.503439 |          0.571538 |        0.352435 |       0.209358 |            4983 |           3567 |

## Multivariate adversarial validation

| variant            |   feature_count |   adversarial_auc |   valid_rows |
|:-------------------|----------------:|------------------:|-------------:|
| all_except_id      |              26 |          1        |        54466 |
| no_decision_day    |              25 |          0.991153 |        54466 |
| no_day_cb_offered  |              23 |          0.797992 |        54466 |
| no_day_rate_limits |              21 |          0.760151 |        54466 |

## Top adversarial importances by variant

### all_except_id

| variant       | feature                           |   importance |
|:--------------|:----------------------------------|-------------:|
| all_except_id | decision_day                      |  0.592993    |
| all_except_id | cb_rate                           |  0.274853    |
| all_except_id | offered_rate                      |  0.0660306   |
| all_except_id | overdraft_limit_max               |  0.0241403   |
| all_except_id | overdraft_limit_min               |  0.0128315   |
| all_except_id | cnt_deb_ul_ip_30                  |  0.00789577  |
| all_except_id | cnt_deb_ul_ip_90                  |  0.00498751  |
| all_except_id | db_group_last                     |  0.00329323  |
| all_except_id | app_term_mean_360                 |  0.00245346  |
| all_except_id | count_all_corp_dashboard_events   |  0.00182253  |
| all_except_id | sum_deb_ul_90                     |  0.00168435  |
| all_except_id | days_from_authperson_registration |  0.00134468  |
| all_except_id | sum_deb_ul_30                     |  0.00126187  |
| all_except_id | fl_adminarea                      |  0.000943929 |
| all_except_id | cnt_deb_loan_90                   |  0.000762618 |
| all_except_id | overdraft_app_term_max_360        |  0.000706089 |
| all_except_id | fl_hdb_bki_total_active_products  |  0.000511096 |
| all_except_id | sum_deb_investment_90             |  0.000375724 |
| all_except_id | balance_rur_amt_30_min            |  0.000243634 |
| all_except_id | loan_rev_min_start_fin            |  0.000197838 |

### no_decision_day

| variant         | feature                           |   importance |
|:----------------|:----------------------------------|-------------:|
| no_decision_day | cb_rate                           |  0.614067    |
| no_decision_day | offered_rate                      |  0.170479    |
| no_decision_day | overdraft_limit_max               |  0.0832054   |
| no_decision_day | overdraft_limit_min               |  0.0438384   |
| no_decision_day | cnt_deb_ul_ip_30                  |  0.0251285   |
| no_decision_day | cnt_deb_ul_ip_90                  |  0.0129869   |
| no_decision_day | db_group_last                     |  0.0072429   |
| no_decision_day | app_term_mean_360                 |  0.00682252  |
| no_decision_day | sum_deb_ul_90                     |  0.00675587  |
| no_decision_day | sum_deb_ul_30                     |  0.00513146  |
| no_decision_day | count_all_corp_dashboard_events   |  0.00453743  |
| no_decision_day | cnt_deb_loan_90                   |  0.00363834  |
| no_decision_day | days_from_authperson_registration |  0.00301246  |
| no_decision_day | fl_adminarea                      |  0.00251164  |
| no_decision_day | fl_hdb_bki_total_active_products  |  0.00201405  |
| no_decision_day | loan_rev_min_start_fin            |  0.00167944  |
| no_decision_day | corp_list                         |  0.00135618  |
| no_decision_day | overdraft_app_term_max_360        |  0.00124451  |
| no_decision_day | sum_deb_investment_90             |  0.00121203  |
| no_decision_day | p75_time_spent_minutes            |  0.000899138 |

### no_day_cb_offered

| variant           | feature                           |   importance |
|:------------------|:----------------------------------|-------------:|
| no_day_cb_offered | overdraft_limit_max               |   0.289596   |
| no_day_cb_offered | overdraft_limit_min               |   0.190389   |
| no_day_cb_offered | cnt_deb_ul_ip_30                  |   0.140878   |
| no_day_cb_offered | cnt_deb_ul_ip_90                  |   0.0748193  |
| no_day_cb_offered | count_all_corp_dashboard_events   |   0.046152   |
| no_day_cb_offered | db_group_last                     |   0.0396374  |
| no_day_cb_offered | app_term_mean_360                 |   0.029395   |
| no_day_cb_offered | sum_deb_ul_30                     |   0.0285481  |
| no_day_cb_offered | sum_deb_ul_90                     |   0.0273733  |
| no_day_cb_offered | cnt_deb_loan_90                   |   0.0204919  |
| no_day_cb_offered | days_from_authperson_registration |   0.0174328  |
| no_day_cb_offered | corp_list                         |   0.0160215  |
| no_day_cb_offered | sum_deb_investment_90             |   0.0134085  |
| no_day_cb_offered | loan_rev_min_start_fin            |   0.0130401  |
| no_day_cb_offered | fl_hdb_bki_total_active_products  |   0.0124261  |
| no_day_cb_offered | overdraft_app_term_max_360        |   0.0109416  |
| no_day_cb_offered | balance_rur_amt_30_min            |   0.0071448  |
| no_day_cb_offered | loan_amount_last                  |   0.00625103 |
| no_day_cb_offered | p75_time_spent_minutes            |   0.00564433 |
| no_day_cb_offered | cnt_cred_loan_90                  |   0.00429429 |

### no_day_rate_limits

| variant            | feature                           |   importance |
|:-------------------|:----------------------------------|-------------:|
| no_day_rate_limits | cnt_deb_ul_ip_30                  |   0.224503   |
| no_day_rate_limits | cnt_deb_ul_ip_90                  |   0.132748   |
| no_day_rate_limits | sum_deb_ul_90                     |   0.0809134  |
| no_day_rate_limits | count_all_corp_dashboard_events   |   0.079513   |
| no_day_rate_limits | sum_deb_ul_30                     |   0.077605   |
| no_day_rate_limits | db_group_last                     |   0.0729123  |
| no_day_rate_limits | app_term_mean_360                 |   0.0590479  |
| no_day_rate_limits | cnt_deb_loan_90                   |   0.0403083  |
| no_day_rate_limits | days_from_authperson_registration |   0.0373623  |
| no_day_rate_limits | sum_deb_investment_90             |   0.0357218  |
| no_day_rate_limits | corp_list                         |   0.0314999  |
| no_day_rate_limits | fl_hdb_bki_total_active_products  |   0.0301204  |
| no_day_rate_limits | loan_rev_min_start_fin            |   0.0256323  |
| no_day_rate_limits | overdraft_app_term_max_360        |   0.0222869  |
| no_day_rate_limits | p75_time_spent_minutes            |   0.0128292  |
| no_day_rate_limits | balance_rur_amt_30_min            |   0.0115508  |
| no_day_rate_limits | cnt_cred_loan_90                  |   0.0111251  |
| no_day_rate_limits | loan_rev_max_start_non_fin        |   0.00522021 |
| no_day_rate_limits | fl_adminarea                      |   0.00460799 |
| no_day_rate_limits | loan_amount_last                  |   0.00229134 |

## Interpretation

- Train/test drift is severe; `decision_day`, `cb_rate`, `offered_rate`, and limit features are major period proxies.
- Fold3 / last-period holdout should be treated as the main model-selection score; OOF remains secondary.
- Current rolling time folds have zero context overlap under the existing context signature, but sibling-offer risk remains relevant for non-temporal CV.
- Feature/model escalation should use small ablations and report Fold3 impact before trusting average OOF gains.

## Validation

- Achieved level: L4 partial
- Checked: time-fold context overlap, repeated context by month, univariate drift, multivariate adversarial train-vs-test drift.
- Remaining: saved alternative group-aware model CV, seed variance, and full red-team review before upload.
