.PHONY: test

# Run tests across all workspace members (all sub-crates).
test:
	cargo test --workspace
