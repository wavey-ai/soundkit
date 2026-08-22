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

.PHONY: container-corpus
container-corpus:
	./scripts/fetch-container-conformance-corpus.sh

.PHONY: container-corpus-test
container-corpus-test: container-corpus
	cargo run --release -p soundkit-container-bench --bin container-corpus -- \
		build/container-corpus scripts/container-corpus-manifest.tsv

.PHONY: codec-fate-corpus
codec-fate-corpus:
	./scripts/fetch-ffmpeg-fate-codec-corpus.sh

.PHONY: codec-fate-test
codec-fate-test: codec-fate-corpus
	cargo run --release -p soundkit-codec-fate --bin soundkit-codec-fate -- \
		check build/fate-codec-corpus scripts/ffmpeg-fate-codec-manifest.tsv

.PHONY: media-pcm-fixtures
media-pcm-fixtures:
	cargo run --release -p soundkit-codec-fate --bin soundkit-codec-fate -- \
		check testdata scripts/media-pcm-fixture-manifest.tsv

.PHONY: media-audio-fuzz
media-audio-fuzz:
	cargo run --release -p soundkit-codec-fate --bin audio-fuzz -- \
		testdata scripts/media-pcm-fixture-manifest.tsv

.PHONY: media-metadata-fate
media-metadata-fate:
	cargo run --release -p soundkit-codec-fate --bin metadata -- \
		check build/ffmpeg-fate-suite scripts/media-metadata-fate-manifest.tsv

.PHONY: media-metadata-sweep
media-metadata-sweep:
	cargo run --release -p soundkit-codec-fate --bin metadata -- \
		sweep build/ffmpeg-fate-suite

.PHONY: codec-fate-bench
codec-fate-bench: codec-fate-corpus
	cargo run --release -p soundkit-codec-fate --bin soundkit-codec-fate -- \
		bench build/fate-codec-corpus scripts/ffmpeg-fate-codec-manifest.tsv \
		$(or $(CODEC_FATE_BENCH_ITERATIONS),5)

.PHONY: container-bench
container-bench:
	cargo run --release -p soundkit-container-bench --bin soundkit-container-bench

.PHONY: container-bench-wasm
container-bench-wasm: wasm
	node scripts/benchmark-container-wasm.mjs

.PHONY: container-bench-all
container-bench-all: container-bench container-bench-wasm
