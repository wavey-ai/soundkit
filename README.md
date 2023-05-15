# soundkit

A package for working with low-latency audio via SRT and via HTTP Range requests

## Audio Packet Header Encoding

This package provides encoding and decoding for audio packet headers. The header encoding is used to specify the details about the audio data that follows in the packet. This includes information about the encoding method, configuration parameters, and frame sizes for each channel.

### Header Structure

The header structure is as follows:

    |-----------------------------------------------------------------------------|
    |    Flag + Config ID    |   Channel Count   |    Frame Size 1    |    ...    |
    |-----------------------------------------------------------------------------|
    |         1 Byte         |       1 Byte      |       2 Bytes      |    ...    |
    |-----------------------------------------------------------------------------|

- **Flag + Config ID**: This is a single byte where the top 3 bits represent the encoding method and the lower 5 bits represent the audio configuration.
- **Channel Count**: A byte that represents the number of channels in the audio data.
- **Frame Size**: There are 2 bytes per frame size. The number of these entries in the header is equal to the channel count. Each entry represents the size of the frame for the corresponding channel.

#### EncodingFlag

The `EncodingFlag` indicates the method of encoding used for the audio data. Currently, two encoding methods are supported:

- `PCM` (value `0`): This flag indicates that Pulse Code Modulation (PCM) encoding is used. PCM is a method used to digitally represent analog signals.
- `Opus` (value `1`): This flag indicates that Opus encoding is used. Opus is a lossy audio coding format designed to efficiently code speech and general audio in a single format.

#### AudioConfig

The `AudioConfig` is a enumeration that represents the sample rate and bit depth of the audio data. The following configurations are supported:

- `Hz44100Bit16`
- `Hz44100Bit24`
- `Hz44100Bit32`
- `Hz48000Bit16`
- `Hz48000Bit24`
- `Hz48000Bit32`
- `Hz88200Bit16`
- `Hz88200Bit24`
- `Hz88200Bit32`
- `Hz96000Bit16`
- `Hz96000Bit24`
- `Hz96000Bit32`
- `Hz176400Bit16`
- `Hz176400Bit24`
- `Hz176400Bit32`
- `Hz192000Bit16`
- `Hz192000Bit24`
- `Hz192000Bit32`
- `Hz352800Bit16`
- `Hz352800Bit24`
- `Hz352800Bit32`

Each configuration indicates a combination of the sample rate (in Hz) and the bit depth (in Bits). For instance, `Hz44100Bit16` represents a sample rate of 44100 Hz and a bit depth of 16 bits.
