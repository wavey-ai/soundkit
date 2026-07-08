.PHONY: test

# Run tests across all workspace members (all sub-crates).
test:
	cargo test --workspace

.PHONY: wasm
wasm:
	wasm-pack build soundkit-wasm-decoder \
		--target web \
		--out-dir pkg \
		--features ogg-opus,webm-opus
