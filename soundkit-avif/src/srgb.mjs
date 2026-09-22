// jSquash 2.1.1 writes 2/2/6 CICP (unspecified primaries/transfer, BT.601
// YUV matrix). Our inputs are explicitly sRGB. Change only the two colour
// descriptors, in BOTH nclx and the AV1 sequence header; preserve matrix,
// range, payload sizes and offsets. No image samples are re-encoded.
// This deliberately accepts only our encoder's reduced still-picture 4:4:4
// headers, failing if a future encoder changes that contract.
// Syntax: AOM AV1 §5.5.1/§5.5.2 and AV1-ISOBMFF §2.3.4.
export function tagAvifSRGB(input) {
    const bytes = new Uint8Array(input), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const type = offset => String.fromCharCode(...bytes.subarray(offset, offset + 4));
    let containers = 0, sequences = 0;
    const fail = () => { throw new Error('The AVIF encoder returned an unsupported colour header.'); };
    function sequence(start, end) {
        let bit = start * 8;
        const read = count => {
            if (bit + count > end * 8) return fail();
            let result = 0;
            for (let i = 0; i < count; i++, bit++) result = result * 2 + ((bytes[bit >> 3] >> (7 - (bit & 7))) & 1);
            return result;
        };
        const profile = read(3);
        if (profile !== 1) return; // An optional monochrome alpha plane has no colour interpretation.
        if (read(1) !== 1 || read(1) !== 1) fail(); // still_picture, reduced_still_picture_header
        read(5); // seq_level_idx[0]
        const widthBits = read(4) + 1, heightBits = read(4) + 1;
        read(widthBits); read(heightBits);
        read(6); // superblock, intra filters, superres, cdef, restoration
        if (read(1) !== 1 || read(1) !== 1) fail(); // high_bitdepth, colour_description_present
        const position = bit;
        const primaries = read(8), transfer = read(8), matrix = read(8);
        if (![1, 2].includes(primaries) || ![2, 13].includes(transfer) || matrix !== 6) fail();
        function write(offset, value) {
            for (let i = 0; i < 8; i++) {
                const index = offset + i, mask = 1 << (7 - (index & 7));
                bytes[index >> 3] = (bytes[index >> 3] & ~mask) | ((value >> (7 - i) & 1) ? mask : 0);
            }
        }
        write(position, 1); write(position + 8, 13); sequences++;
    }
    function obus(start, end) {
        let offset = start;
        while (offset < end) {
            const header = bytes[offset++], kind = (header >> 3) & 15;
            if ((header & 0x81) || !(header & 2)) fail();
            if (header & 4) offset++;
            let size = 0, shift = 0, more;
            do {
                if (offset >= end || shift > 49) fail();
                const value = bytes[offset++]; size += (value & 127) * 2 ** shift; shift += 7; more = value & 128;
            } while (more);
            if (offset + size > end) fail();
            if (kind === 1) sequence(offset, offset + size);
            offset += size;
        }
    }
    function boxes(start, end) {
        let offset = start;
        while (offset < end) {
            if (offset + 8 > end) fail();
            const size = view.getUint32(offset), name = type(offset + 4), body = offset + 8;
            if (size < 8 || offset + size > end) fail();
            if (name === 'meta') boxes(body + 4, offset + size);
            else if (name === 'iprp' || name === 'ipco') boxes(body, offset + size);
            else if (name === 'colr' && type(body) === 'nclx') {
                if (size !== 19 || view.getUint16(body + 8) !== 6) fail();
                view.setUint16(body + 4, 1); view.setUint16(body + 6, 13); containers++;
            } else if (name === 'mdat') obus(body, offset + size);
            else if (name === 'av1C' && size > 12) obus(body + 4, offset + size);
            offset += size;
        }
    }
    boxes(0, bytes.length);
    if (!containers || !sequences) fail();
    return bytes;
}
