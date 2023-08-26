build:
	cargo build

.PHONY: test
test:
	cargo test -- --nocapture

wasm:
	wasm-pack build --no-default-features --target no-modules --no-typescript --out-dir pkg --debug
