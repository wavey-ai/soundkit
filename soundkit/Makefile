build:
	cargo build

.PHONY: test
test:
	cargo test -- --nocapture

wasm:
	wasm-pack build --features wasm --no-default-features --target wasm32-unknown-unknown
