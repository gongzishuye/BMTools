
# Summary
## Offline
- batch_size = 1
- Offline
- PerformanceOnly
- 三个三芯卡

```
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 1.21305
Result is : INVALID
  Min duration satisfied : NO
  Min queries satisfied : Yes
Recommendations:
 * Increase expected QPS so the loadgen pre-generates a larger (coalesced) query.
```


## SingleStream
- batch_size = 1
- SingleStream
- PerformanceOnly
- 三个三芯卡


```
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : SingleStream
Mode     : PerformanceOnly
90th percentile latency (ns) : 8434934462
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
```