# Windows output replacement repair — 2026-10-15

The 8% run stopped after 171 valid attempts (19 complete frames), with WinError 5
while atomically replacing outcomes.csv. The launcher's periodic CSV reader could
deny replacement on Windows. The exact conflicting handle was not captured.

Changes: progress now reads completed frame messages from the append-only log;
atomic CSV/JSON replacement retries Windows errors 5/32/33 for up to five seconds.
Persistent errors still fail without replacing the previous valid results.

Validation: 23 regression tests passed using the workspace temporary directory.
An actual Windows open-reader test recovered after the reader released its handle
(0.76 seconds), with both rows intact. The first suite invocation failed because
its sandbox temporary directory denied access; rerunning with workspace TEMP/TMP
passed. No numerical model or sampling settings changed.

The original failed run is preserved as results/local/run_8pct_seed20261014_failed_20261015.
The corrected workflow restarts the identical selection at the usual default path.
It intentionally recomputes the 19 frames so that code fingerprints remain strict;
no original manifest is edited to make old results appear newly generated.
