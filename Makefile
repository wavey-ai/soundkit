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
	node scripts/test-video-wasm.mjs
