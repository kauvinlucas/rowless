# rowless-iceberg

Iceberg schema source, via PyIceberg catalog metadata.

## The trick

Read the table metadata, never the data files. Iceberg keeps stable field IDs
across renames and retains schema history, so this is the one source where a
rename is a recorded fact rather than a heuristic, and a diff can run against a
historical snapshot without a git checkout.

## Status

Skeleton. Scheduled after the first release, alongside PySpark.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
