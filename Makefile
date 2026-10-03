# Keep Cargo invocations serial, including when make is called with -j.
.NOTPARALLEL:
.DEFAULT_GOAL := help

CARGO ?= cargo
# The default suite needs RocksDB build tools, but no FoundationDB client.
PACKAGES ?= --workspace --exclude ktann-foundationdb
PROFILE ?= smoke
REPORT_DIR ?= .benchmark-data/results
BENCH_ARGS ?=

.PHONY: help build release check test fmt fmt-check lint doc verify verify-all test-fdb bench bench-fdb

help:
	@printf '%s\n' \
	  'KTANN development commands' \
	  '  build / release  Build without FoundationDB (debug / optimized)' \
	  '  check            Check all default-package targets' \
	  '  test             Run tests without FoundationDB' \
	  '  fmt / fmt-check  Format Rust / check formatting' \
	  '  lint             Run Clippy with warnings denied' \
	  '  doc              Build API documentation' \
	  '  verify           Check formatting, lint, and test in sequence' \
	  '  verify-all       Verify all packages and features (needs libfdb_c)' \
	  '  test-fdb         Run integration tests against a local FDB cluster' \
	  '  bench / bench-fdb Run optimized RocksDB / FoundationDB benchmarks' \
	  '' \
	  'Overrides: CARGO, PACKAGES, PROFILE=smoke|full|large, REPORT_DIR, BENCH_ARGS' \
	  'Example: make bench PROFILE=full BENCH_ARGS="--scenario import-to-search-lifecycle"' \
	  'See adapter READMEs for native dependencies and cluster setup.'

build:
	$(CARGO) build $(PACKAGES)

release:
	$(CARGO) build $(PACKAGES) --release

check:
	$(CARGO) check $(PACKAGES) --all-targets

test:
	$(CARGO) test $(PACKAGES)

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy $(PACKAGES) --all-targets -- -D warnings

doc:
	$(CARGO) doc $(PACKAGES) --no-deps

verify: fmt-check lint test

verify-all: fmt-check
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings
	$(CARGO) test --workspace --all-features

# Durability tests require a server restart; follow the adapter README separately.
test-fdb:
	$(CARGO) test -p ktann-foundationdb \
	  --test foundationdb_adapter --test foundationdb_faults \
	  --test foundationdb_verify --test foundationdb_observability \
	  --test foundationdb_recall -- --ignored

bench:
	@mkdir -p "$(REPORT_DIR)"
	$(CARGO) run --release -p ktann-benchmarks --bin ktann-bench -- \
	  run --backend rocksdb --profile $(PROFILE) \
	  --output "$(REPORT_DIR)/rocksdb-$(PROFILE).json" $(BENCH_ARGS)

bench-fdb:
	@mkdir -p "$(REPORT_DIR)"
	$(CARGO) run --release -p ktann-benchmarks --no-default-features \
	  --features foundationdb --bin ktann-bench -- \
	  run --backend foundationdb --profile $(PROFILE) \
	  --output "$(REPORT_DIR)/foundationdb-$(PROFILE).json" $(BENCH_ARGS)
