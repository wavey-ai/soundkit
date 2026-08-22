# External container corpus

SoundKit can run upstream conformance files without committing large media
assets to this repository:

```sh
make container-corpus-test
```

The fetch step downloads the official eight-file
[Matroska test suite](https://www.matroska.org/downloads/test_suite.html) at
revision `e6965e5ca666322ed93e2748a10a4f132309e005`. Each file is verified
against the SHA-256 recorded in
`scripts/fetch-container-conformance-corpus.sh` and stored below the ignored
`build/container-corpus` directory.

The release runner sends every file through SoundKit at 4 KiB, 64 KiB, and
4 MiB input sizes. Accepted files must produce byte-identical event
fingerprints, packet counts, and payload totals at every size. Expected
rejections—whether malformed or outside SoundKit's declared codec/container
contract—must reject at every size. Each parser run also has a 128 MiB
peak-live allocation budget above the already-loaded source bytes.

`scripts/container-corpus-manifest.tsv` records expected acceptance. Keep an
unsupported codec separate from a malformed container: the container should
still parse and expose its track metadata when its structure is valid.

Broader local sweeps can add files under the same ignored corpus root and
append manifest rows. Recommended sources are:

- FFmpeg FATE for MOV/MP4, fragmented MP4, Ogg, MPEG-TS/M2TS, CAF, MXF, and
  DNx regression files.
- The MPEG File Format Conformance Framework for ISO-BMFF box coverage.
- FADGI and MediaConch sample sets for well-formed and deliberately malformed
  preservation MXF.

Before adding a source, record an immutable upstream corpus revision and every
file hash, then confirm that its license permits redistribution or local
automated download.
