# Container performance baseline

Date: 2026-08-22

This baseline uses an Apple M1 MacBook Air with 16 GB of memory.

The native build uses Rust 1.96.0. Run `make container-bench` to repeat the test.

The benchmark discards public packet events after each push. It measures parser throughput and allocation activity.

Peak bytes are the maximum additional live bytes during one parse. The custom allocator measures this value.

| Container | Push size | MiB/s | Allocations | Peak bytes | Events |
| --- | ---: | ---: | ---: | ---: | ---: |
| fMP4 | 1 B | 0.37 | 11,011,675 | 152,639 | 143 |
| fMP4 | 188 B | 41.35 | 68,500 | 152,639 | 143 |
| fMP4 | 4 KiB | 424.50 | 5,874 | 152,639 | 143 |
| fMP4 | 64 KiB | 1,652.23 | 981 | 240,288 | 143 |
| fMP4 | 4 MiB | 1,650.08 | 837 | 564,259 | 143 |
| WebM | 1 B | 9.68 | 2,444 | 102,728 | 152 |
| WebM | 188 B | 505.78 | 2,432 | 102,728 | 152 |
| WebM | 4 KiB | 940.45 | 2,037 | 102,728 | 152 |
| WebM | 64 KiB | 1,022.86 | 1,938 | 237,235 | 152 |
| WebM | 4 MiB | 1,043.24 | 1,923 | 411,818 | 152 |
| TS | 1 B | 16.99 | 139 | 14,822 | 49 |
| TS | 188 B | 767.29 | 139 | 14,822 | 49 |
| TS | 4 KiB | 1,198.67 | 141 | 31,494 | 49 |
| TS | 64 KiB | 1,214.17 | 139 | 50,426 | 49 |
| TS | 4 MiB | 946.58 | 139 | 50,426 | 49 |
| M2TS | 1 B | 16.55 | 139 | 14,822 | 49 |
| M2TS | 188 B | 931.81 | 140 | 19,462 | 49 |
| M2TS | 4 KiB | 1,240.08 | 141 | 31,494 | 49 |
| M2TS | 64 KiB | 1,288.95 | 139 | 54,698 | 49 |
| M2TS | 4 MiB | 1,341.68 | 139 | 54,698 | 49 |
| Ogg | 1 B | 14.26 | 216 | 34,901 | 151 |
| Ogg | 188 B | 133.49 | 215 | 51,285 | 151 |
| Ogg | 4 KiB | 149.22 | 215 | 51,285 | 151 |
| Ogg | 64 KiB | 141.67 | 204 | 91,167 | 151 |
| Ogg | 4 MiB | 143.41 | 204 | 91,167 | 151 |
| MP4 | Seekable | 6,536.72 | 254 | 42,115 | 219 |
| CAF | Seekable | 21,002.92 | 16 | 648 | 7 |

The 4 MiB throughput stays above 77% of the 64 KiB result for each streaming container.

One-byte fMP4 pushes have high call and allocation costs. Production callers must use larger bounded reads when possible.

## WebAssembly results

The WebAssembly build uses `wasm-pack 0.15.0` and Node.js 26.3.0.

WASM bytes show the final linear-memory size in a new process. Growth is relative to the initialized module.

| Container | Push size | MiB/s | WASM bytes | Growth bytes | Events |
| --- | ---: | ---: | ---: | ---: | ---: |
| fMP4 | 1 B | 0.41 | 2,162,688 | 196,608 | 143 |
| fMP4 | 188 B | 17.95 | 2,162,688 | 196,608 | 143 |
| fMP4 | 4 KiB | 38.70 | 2,162,688 | 196,608 | 143 |
| fMP4 | 64 KiB | 76.02 | 2,228,224 | 262,144 | 143 |
| fMP4 | 4 MiB | 78.86 | 2,752,512 | 786,432 | 143 |
| WebM | 1 B | 1.66 | 2,097,152 | 131,072 | 152 |
| WebM | 188 B | 27.13 | 2,097,152 | 131,072 | 152 |
| WebM | 4 KiB | 50.75 | 2,097,152 | 131,072 | 152 |
| WebM | 64 KiB | 62.46 | 2,359,296 | 393,216 | 152 |
| WebM | 4 MiB | 58.50 | 2,555,904 | 589,824 | 152 |
| TS | 1 B | 0.80 | 1,966,080 | 0 | 49 |
| TS | 188 B | 8.61 | 1,966,080 | 0 | 49 |
| TS | 4 KiB | 16.10 | 1,966,080 | 0 | 49 |
| TS | 64 KiB | 21.93 | 2,031,616 | 65,536 | 49 |
| TS | 4 MiB | 23.37 | 2,031,616 | 65,536 | 49 |
| M2TS | 1 B | 1.12 | 1,966,080 | 0 | 49 |
| M2TS | 188 B | 7.51 | 1,966,080 | 0 | 49 |
| M2TS | 4 KiB | 14.67 | 1,966,080 | 0 | 49 |
| M2TS | 64 KiB | 25.35 | 2,031,616 | 65,536 | 49 |
| M2TS | 4 MiB | 26.19 | 2,031,616 | 65,536 | 49 |
| Ogg | 1 B | 1.15 | 1,966,080 | 0 | 151 |
| Ogg | 188 B | 10.83 | 2,031,616 | 65,536 | 151 |
| Ogg | 4 KiB | 18.58 | 2,031,616 | 65,536 | 151 |
| Ogg | 64 KiB | 23.71 | 2,031,616 | 65,536 | 151 |
| Ogg | 4 MiB | 24.01 | 2,031,616 | 65,536 | 151 |
| MP4 | Seekable | 608.28 | 2,228,224 | 262,144 | 219 |
| CAF | Seekable | 324.84 | 1,966,080 | 0 | 7 |

The WASM 4 MiB throughput stays above 93% of the 64 KiB result for each streaming container.
