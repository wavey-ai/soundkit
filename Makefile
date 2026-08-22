.PHONY: test

# Run tests across all workspace members (all sub-crates).
test:
	cargo test --workspace

.PHONY: wasm
wasm:
	wasm-pack build soundkit-wasm \
		--target web \
		--out-dir pkg \
		--features default

.PHONY: media-fixtures
media-fixtures:
	@test -n "$(SOURCE_MEDIA)" || (echo "SOURCE_MEDIA must point to a music video" >&2; exit 2)
	./scripts/generate-video-compat-fixtures.sh "$(SOURCE_MEDIA)"

.PHONY: media-conformance
media-conformance: wasm
	cd testdata/video-compat/never-final && shasum -a 256 -c SHA256SUMS
	node scripts/test-video-wasm.mjs

.PHONY: media-upstream-corpus
media-upstream-corpus:
	./scripts/fetch-video-conformance-corpus.sh

.PHONY: media-upstream-conformance
media-upstream-conformance: wasm media-upstream-corpus
	node scripts/test-video-upstream-wasm.mjs

.PHONY: media-fuzz
media-fuzz: wasm media-upstream-corpus
	node scripts/fuzz-video-wasm.mjs

.PHONY: container-bench
container-bench:
	cargo run --release -p soundkit-container-bench

.PHONY: container-bench-wasm
container-bench-wasm: wasm
	node scripts/benchmark-container-wasm.mjs

.PHONY: container-bench-all
container-bench-all: container-bench container-bench-wasm
