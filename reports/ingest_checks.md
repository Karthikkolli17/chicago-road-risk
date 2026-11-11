# Ingest checks

Generated 2026-11-11 11:02.

- **Road segments (motorways excluded):** 48696

## Crashes per year

|   crash_year |   crashes |   ksi |   with_pedestrian |   with_cyclist |   pct_bad_coords |
|-------------:|----------:|------:|------------------:|---------------:|-----------------:|
|         2021 |    108767 |  2187 |              2382 |           1561 |             0.75 |
|         2022 |    108413 |  2093 |              2598 |           1717 |             0.87 |
|         2023 |    110754 |  2035 |              2882 |           1973 |             0.94 |
|         2024 |    112056 |  1739 |              3183 |           2339 |             1.07 |
|         2025 |    109116 |  1573 |              3042 |           2521 |             1.06 |
|         2026 |     96642 |  1396 |              2700 |           2572 |             1.16 |

## Coordinate quality

Crashes not `ok` are kept in the warehouse but cannot be snapped to a road.

| coord_status   |   crashes |   ksi |   pct |
|:---------------|----------:|------:|------:|
| ok             |    639476 | 10936 | 99.03 |
| missing        |      6227 |    86 |  0.96 |
| zero           |        45 |     1 |  0.01 |

## People rows

|   vru_people_without_crash |
|---------------------------:|
|                          0 |
