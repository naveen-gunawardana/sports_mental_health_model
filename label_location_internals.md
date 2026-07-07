# `label_location` Internals

This page documents the operational details of the `label_location` resource: its persistent caches, concurrency story, year-band scan knobs, per-post deduplication, and memory/disk budget. For the high-level role of `label_location` in the pipeline, see the [main README](README.md). For the meaning of `location` / `location_prob` / `contender_location` columns in the output, see [variable_list.md](variable_list.md).

## How the resource estimates a location

For each author appearing in the curated input month, `label_location`:

1. **Pass 1 — Curated scan.** Reads the month's curated CSV once to identify the set of authors whose rows still need writing, plus a per-author "local activity" count.
2. **Pass 2 — Curated seed.** Re-reads the curated CSV and accumulates per-author feature counts (`word:tok`, `subreddit:name`, `hour:HH`) from the group-relevant posts. Also collects the post IDs that contributed, for later deduplication.
3. **Pass 3 — Raw spiral scan.** Opens the raw `.zst` files for a spiral of months centered on the target month (see [Scan-spiral knobs](#scan-spiral-knobs-and-year-band-defaults)) and accumulates the author's broader Reddit activity from `data/data_reddit_raw/{type}/`. The raw scan reads `comments` regardless of the input type, as comments made up more than 90% of the training data.
4. **Inference.** A weighted mixture of logistic regressions over word, subreddit, and hour features produces tiered predictions: first US vs Non-US, then state (for US) or region (for Non-US). Authors whose top probability falls below level-specific confidence thresholds are labeled `UNK`. Defaults can be changed inside `label_location`.
5. **Write.** Each curated row is streamed to the output CSV with `location`, `location_prob`, `contender_location`, and `contender_location_prob` appended.

## Persistent caches

`label_location` maintains a SQLite database at `data/data_reddit_curated/data_reddit_location/author_location_cache_{type}.sqlite`. Here, `type` is the raw feature source (`comments` by default), not the curated input type. As a result, **all six social groups share the same cache** when run with the same raw type: running sexuality/comments first populates a cache that race/comments, age/comments, etc. then read from for free.

The database holds two tables:

- **`author_location` + `author_location_detail`**: Final-decision cache. One row per author with the inferred location, top probabilities, tier (`top` / `region` / `state`), and contender. Cache hits here bypass Pass 3 entirely for that author. Confidence-aware upsert: a new write only replaces an existing row when its probability is strictly greater.
- **`author_file_counts`**: Scan-progress cache. One row per `(author, raw_file)` tuple **where the author actually appeared**, storing the zstd-compressed per-author feature counts and per-author seen-count contributed by that file. Files the scanner opened but where the author didn't appear do not produce rows; the storage cost would have been disproportionate to the rare cases where an entire spiral file's targets were all cached.

## Scan-state amortization

When `label_location` starts on a new month, it consults `author_file_counts` for every author it needs to scan. For each author, the spiral's basenames are subtracted from the basenames the cache already has them in, yielding the per-author set of files that still need scanning. The scanner is then driven over only the union of those needed files.
The target month's `.zst` file is an exception: it is **always rescanned** so the per-post dedup pass (described below) can run against the current run's curated IDs.
Because the cache records only "author was found" rows, not "scanner opened the file for this author," next month's overlapping spiral may re-target authors at files where they had no prior hits. The file gets opened anyway for any other author who needs it, so the marginal cost is just a fast per-line author-set lookup.

## Concurrency

Both caches are safe under any concurrency level:

- The **final-decision cache** uses confidence-aware upsert, so concurrent writers cannot lose information as the more confident label always wins.
- The **scan-progress cache** uses `INSERT OR IGNORE` keyed by `(author, raw_file)`. Two array tasks scanning the same `(author, file)` tuple do byte-for-byte identical work (same raw `.zst`, same logic), so whichever row commits first wins and the duplicate is dropped silently. The redundant scan is wasted compute but never corrupting.

There is no need to serialize `label_location` array tasks. Both `--num-jobs 1` and higher concurrency produce identical labels.

Operationally, the choice between parallel and serial is a wall-clock-vs-compute trade-off on a **cold** cache:

- **Parallel** (`--num-jobs N > 1`) minimizes wall-clock time. Tasks with heavily overlapping spirals will independently scan the same raw files; the scans converge correctly via `INSERT OR IGNORE` but the cluster CPU/GPU hours are higher.
- **Serial** (`--num-jobs 1`) maximizes cache amortization on a cold dataset: each subsequent month hits the cache for ~90%+ of the prior month's spiral and finishes faster. Lower total cluster usage at the cost of wall time.

Either way, the second pass over the same corpus much faster than the first because both caches are populated. To the extent of author overlap, the benefits carry over to other `group` runs. 

## Per-post deduplication

The target month's `.zst` file is a superset of the curated CSV's content (the curated CSV is a group-keyword-filtered subset of the raw posts). Without deduplication, Pass 2 (curated seed) and Pass 3 (raw spiral) would count the same post's features twice when they both encounter it — once as a curated row, once as a raw `.zst` row.
The dedup pass eliminates this:
1. Pass 2 records `(author, post_id)` for every curated row it ingests.
2. Pass 3, while scanning the target month's `.zst`, tracks two count dicts per author: the full raw counts (cached, group-agnostic) and a side overlap counts dict for any post whose ID matches one Pass 2 recorded.
3. At inference time, the combined input is `local_counts + (raw_counts − overlap_counts)`. Each post is counted exactly once.
The persisted `author_file_counts` rows store the **undeduplicated** raw counts so that a different social group's later run (with a different curated ID set) can compute its own group-specific overlap subtraction.
The dedup is only meaningful when the curated input type matches the raw type (the comments-on-comments case). For submissions-on-comments runs, the curated CSV and the raw `.zst` files contain different post IDs by construction, and no double-count is possible.

## Scan-spiral knobs and year-band defaults

The spiral around the target month is governed by three command-line flags:
- `--maxitems` — Maximum raw posts per author for the whole scan, **cumulative across every raw file visited**. Matches the sampling distribution the LR model was trained on. When the scan-state cache already holds samples for an author from prior runs, the effective remaining quota for the current run is `max_items − already_cached_seen`.
- `--maxfiles` — Hard cap on the number of raw files visited per spiral.
- `--maxradius` — Maximum month offset (in either direction) from the target month.

Because raw `.zst` files grow several times larger in recent years, the defaults switch on year:

| Year band | `--maxitems` | `--maxradius` | `--maxfiles` |
|---|---|---|---|
| 2007–2019 (stringent) | 50 | 30 | 60 |
| 2020–2023 (relaxed) | 25 | 20 | 60 |

The relaxation for 2020+ roughly halves per-author sampling cost on the largest months without materially shifting inference quality. CLI flags override the band defaults when set.

## Memory and disk budget

**Per-month peak RAM** scales with the month's raw volume because of:
- The curated CSV row pass.
- Per-author count dictionaries during Pass 2 and Pass 3 aggregation.
- The bundle of pickled logistic-regression models (~few hundred MB resident).
Measured across ISAAC development runs (`comments`), per-task `MaxRSS` had a **median of ~19 GB and a maximum of ~35 GB**; the heaviest tasks are the high-volume months from 2018 onward, while small early-corpus months use only a few GB. Size `--mem` to the months you are processing: a `--mem 16G` request is enough for the stringent-band early-corpus months but OOMs on the high-volume later months — request `--mem 24G` (or more) for tasks covering 2018 and onward.

**Persistent cache disk usage** is bounded by the cumulative cap: each author has at most `--maxitems` samples across the entire corpus regardless of how many months they appear in. Combined with only persisting rows where the author was actually found:
- ~100–300 bytes per `author_file_counts` row after zstd compression.
- ~5–30 rows per author on average, depending on how thinly their samples are spread.
- For a ~2M-author corpus: **~1–5 GB of SQLite data** at full saturation.
The cache lives on the shared filesystem and is reused across runs and groups, so the storage cost is paid once.

## Crash recovery and resume

The output is written in source_row order via an incremental flush every `FLUSH_EVERY_N_RAW = 3` raw files scanned. When an author hits the per-spiral sampling cap, their rows are inferred and written immediately. A task killed mid-scan leaves a partial output CSV whose `source_row` column is monotonically increasing.
On resubmission, `get_last_source_row` reads the partial output and the resource resumes from the first unwritten row. The `author_file_counts` cache holds whatever was scanned before the crash, so the resumed task does not re-scan files already covered. See [the Resumable Runs section of the main README](README.md#resumable-runs-and-the-source_row-column) for the cross-resource source_row resume mechanism.

## Known limitations

- **The dedup pass forces a rescan of the target month's `.zst` even when fully cached.** This is ~3% of the spiral's I/O and is the price of the dedup correction. Other spiral files retain full cache benefit.
