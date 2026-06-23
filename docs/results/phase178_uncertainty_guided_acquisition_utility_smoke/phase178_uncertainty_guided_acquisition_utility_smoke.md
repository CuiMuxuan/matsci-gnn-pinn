# Phase 178 Uncertainty-Guided Acquisition Utility Smoke

## Gate
- Status: `phase178_uncertainty_guided_acquisition_smoke_closed_no_guarded_acquisition_gain`
- Selected policy: `uniform_budget_control`
- Best candidate policy: `hybrid_uncertainty_hot_gradient_candidate`
- Best control policy: `uniform_budget_control`
- Validation utility gain vs best control: `-1.136014204`
- Phase 179 training design allowed: `false`
- Phase 178 model training allowed: `false`
- Phase 179 training allowed now: `false`
- A100 training allowed now: `false`
- A100-SXM4-80GB request now: `false`

## Interpretation
This no-training smoke tests acquisition utility only. If a same-budget uniform/random/no-new control wins validation selection, the uncertainty acquisition route closes before any model training.

## Policies
| policy_id | family | executed | is_control | description |
| --- | --- | --- | --- | --- |
| posterior_entropy_reduction_candidate | acquisition_candidate | true | false | selects points with high posterior predictive variance |
| latent_ensemble_disagreement_candidate | acquisition_candidate | true | false | selects points with high top-latent ensemble disagreement |
| hybrid_uncertainty_hot_gradient_candidate | acquisition_candidate | true | false | combines posterior variance with a fixed hot-gradient quota proxy |
| uniform_budget_control | control | true | true | same-budget uniform candidate-pool acquisition |
| random_budget_control | control | true | true | same-budget deterministic random acquisition |
| no_new_observation_control | control | true | true | posterior update without adding observations |

## Summary Metrics
| policy_id | family | split | seed_count | case_count | posterior_trace_contraction_mean | parameter_error_gain_mean | closure_abs_error_gain_mean | parameter_error_after_mean | closure_abs_error_after_mean | duplicate_fraction_mean | boundary_fraction_mean | utility_score_mean | utility_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_uncertainty_hot_gradient_candidate | acquisition_candidate | val | 3 | 48 | -0.020898624 | -0.5119685927 | -0.1138969877 | 0.7865684689 | 0.1728351497 | 0 | 0.0052083333 | -1.031772035 | 0.8556594788 |
| hybrid_uncertainty_hot_gradient_candidate | acquisition_candidate | test | 3 | 48 | 0.0110117759 | -0.3900705846 | -0.0862684248 | 0.7984952152 | 0.1754305897 | 0 | 0.0052083333 | -0.7810986106 | 0.9171244104 |
| latent_ensemble_disagreement_candidate | acquisition_candidate | val | 3 | 48 | -0.0375263458 | -0.6745922453 | -0.1497317989 | 0.9491921215 | 0.2086699609 | 0 | 0.0711805556 | -1.358943966 | 1.355084995 |
| latent_ensemble_disagreement_candidate | acquisition_candidate | test | 3 | 48 | -0.0582916072 | -0.5674230596 | -0.125296685 | 0.9758476902 | 0.2144588499 | 0 | 0.0833333333 | -1.142782607 | 1.29973591 |
| no_new_observation_control | control | val | 3 | 48 | 0 | 0 | 0 | 0.2745998762 | 0.058938162 | 0 | 0 | 0 | 0 |
| no_new_observation_control | control | test | 3 | 48 | 0 | 0 | 0 | 0.4084246306 | 0.0891621649 | 0 | 0 | 0 | 0 |
| posterior_entropy_reduction_candidate | acquisition_candidate | val | 3 | 48 | -0.0227777138 | -0.5467101869 | -0.1214785201 | 0.8213100631 | 0.1804166821 | 0 | 0.1736111111 | -1.10116305 | 1.355836999 |
| posterior_entropy_reduction_candidate | acquisition_candidate | test | 3 | 48 | -0.0379206793 | -0.2965751768 | -0.0656842748 | 0.7049998074 | 0.1548464397 | 0 | 0.2309027778 | -0.5989321302 | 1.326014275 |
| random_budget_control | control | val | 3 | 48 | 0.0183063677 | -0.0864079755 | -0.0195558567 | 0.3610078517 | 0.0784940187 | 0 | 0.2152777778 | -0.1734675963 | 0.4430696834 |
| random_budget_control | control | test | 3 | 48 | 0.0414102698 | 0.0609122756 | 0.0136856786 | 0.347512355 | 0.0754764864 | 0 | 0.2083333333 | 0.1272609324 | 0.4268550611 |
| uniform_budget_control | control | val | 3 | 48 | 0.0132677196 | 0.0507283087 | 0.0114811592 | 0.2238715675 | 0.0474570028 | 0 | 0.25 | 0.1042421681 | 0.3701548408 |
| uniform_budget_control | control | test | 3 | 48 | 0.0231139353 | 0.1118138778 | 0.0247882365 | 0.2966107528 | 0.0643739284 | 0 | 0.25 | 0.2267990735 | 0.3534965144 |

## Seed Summary
| seed | policy_id | split | case_count | posterior_trace_contraction_mean | parameter_error_gain_mean | closure_abs_error_gain_mean | utility_score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 178 | hybrid_uncertainty_hot_gradient_candidate | val | 16 | -0.0313597649 | -0.476691497 | -0.106481467 | -0.9638341418 |
| 178 | hybrid_uncertainty_hot_gradient_candidate | test | 16 | 0.0219189083 | -0.2726392759 | -0.0605137528 | -0.5455098978 |
| 178 | latent_ensemble_disagreement_candidate | val | 16 | -0.0684375051 | -0.6449191223 | -0.143588236 | -1.304436673 |
| 178 | latent_ensemble_disagreement_candidate | test | 16 | 0.0031866235 | -0.5334852506 | -0.1180510349 | -1.069762201 |
| 178 | no_new_observation_control | val | 16 | 0 | 0 | 0 | 0 |
| 178 | no_new_observation_control | test | 16 | 0 | 0 | 0 | 0 |
| 178 | posterior_entropy_reduction_candidate | val | 16 | -0.0482387924 | -0.5313256492 | -0.1184949201 | -1.074762802 |
| 178 | posterior_entropy_reduction_candidate | test | 16 | -0.0015827573 | -0.2123523914 | -0.047300061 | -0.4275109444 |
| 178 | random_budget_control | val | 16 | 0.0092932149 | -0.0610948116 | -0.0141201268 | -0.1243478848 |
| 178 | random_budget_control | test | 16 | 0.041400899 | 0.0764166803 | 0.0172812483 | 0.1591078988 |
| 178 | uniform_budget_control | val | 16 | 0.0077672883 | 0.0524138625 | 0.0118107153 | 0.1068756607 |
| 178 | uniform_budget_control | test | 16 | 0.0187728357 | 0.141212065 | 0.0311518861 | 0.2846888307 |
| 179 | hybrid_uncertainty_hot_gradient_candidate | val | 16 | -0.0219438823 | -0.5794926923 | -0.128458596 | -1.16558979 |
| 179 | hybrid_uncertainty_hot_gradient_candidate | test | 16 | 0.0388591336 | -0.4479587615 | -0.098843386 | -0.8933609665 |
| 179 | latent_ensemble_disagreement_candidate | val | 16 | -0.0153815399 | -0.7484887432 | -0.1656588072 | -1.503021475 |
| 179 | latent_ensemble_disagreement_candidate | test | 16 | -0.0301106478 | -0.5723367509 | -0.1260322623 | -1.148221735 |
| 179 | no_new_observation_control | val | 16 | 0 | 0 | 0 | 0 |
| 179 | no_new_observation_control | test | 16 | 0 | 0 | 0 | 0 |
| 179 | posterior_entropy_reduction_candidate | val | 16 | -0.0094575539 | -0.5970869377 | -0.1321635628 | -1.198776161 |
| 179 | posterior_entropy_reduction_candidate | test | 16 | 0.0128977306 | -0.3659707331 | -0.0807628217 | -0.731784695 |
| 179 | random_budget_control | val | 16 | 0.0248550104 | -0.1139600501 | -0.0256657801 | -0.2281371857 |
| 179 | random_budget_control | test | 16 | 0.0646028192 | 0.0088895681 | 0.0020223249 | 0.0245422361 |
| 179 | uniform_budget_control | val | 16 | 0.0181896913 | 0.0608333606 | 0.0137989819 | 0.1253749747 |
| 179 | uniform_budget_control | test | 16 | 0.0406779858 | 0.0605244849 | 0.0134684955 | 0.1258127177 |
| 180 | hybrid_uncertainty_hot_gradient_candidate | val | 16 | -0.0093922247 | -0.4797215886 | -0.1067509001 | -0.965892175 |
| 180 | hybrid_uncertainty_hot_gradient_candidate | test | 16 | -0.027742714 | -0.4496137165 | -0.0994481355 | -0.9044249676 |
| 180 | latent_ensemble_disagreement_candidate | val | 16 | -0.0287599925 | -0.6303688705 | -0.1399483535 | -1.269373749 |
| 180 | latent_ensemble_disagreement_candidate | test | 16 | -0.1479507973 | -0.5964471772 | -0.1318067579 | -1.210363884 |
| 180 | no_new_observation_control | val | 16 | 0 | 0 | 0 | 0 |
| 180 | no_new_observation_control | test | 16 | 0 | 0 | 0 | 0 |
| 180 | posterior_entropy_reduction_candidate | val | 16 | -0.0106367951 | -0.5117179738 | -0.1137770774 | -1.029950187 |
| 180 | posterior_entropy_reduction_candidate | test | 16 | -0.1250770114 | -0.3114024061 | -0.0689899417 | -0.6375007511 |
| 180 | random_budget_control | val | 16 | 0.0207708779 | -0.0841690647 | -0.0188816631 | -0.1679177184 |
| 180 | random_budget_control | test | 16 | 0.018227091 | 0.0974305784 | 0.0217534625 | 0.1981326624 |
| 180 | uniform_budget_control | val | 16 | 0.013846179 | 0.0389377032 | 0.0088337806 | 0.080475869 |
| 180 | uniform_budget_control | test | 16 | 0.0098909844 | 0.1337050834 | 0.0297443279 | 0.2698956721 |
