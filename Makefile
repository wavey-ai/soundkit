build:
	cargo build
	cargo build --lib --release --target aarch64-apple-darwin

.PHONY: test
test:
	cargo test -- --nocapture
