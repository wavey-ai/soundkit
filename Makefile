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
