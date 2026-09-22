// An original, deterministic 12-bit Bayer capture in a TIFF/DNG container.
export function makeDNG({ width = 320, height = 240, orientation = 1 } = {}) {
    const tags = [];
    const tag = (id, type, values) => {
        let bytes, count;
        if (type === 2) { bytes = Buffer.from(values + '\0'); count = bytes.length; }
        else {
            count = values.length;
            const size = ({ 1: 1, 3: 2, 4: 4, 5: 8, 10: 8 })[type];
            bytes = Buffer.alloc(count * size);
            values.forEach((value, index) => {
                if (type === 1) bytes.writeUInt8(value, index);
                else if (type === 3) bytes.writeUInt16LE(value, index * 2);
                else if (type === 4) bytes.writeUInt32LE(value, index * 4);
                else { bytes.writeInt32LE(Math.round(value * 100000), index * 8); bytes.writeInt32LE(100000, index * 8 + 4); }
            });
        }
        tags.push({ id, type, count, bytes });
    };
    tag(256, 4, [width]); tag(257, 4, [height]); tag(258, 3, [16]); tag(259, 3, [1]); tag(262, 3, [32803]);
    tag(271, 2, 'SoundKit'); tag(272, 2, 'Synthetic Bayer'); tag(273, 4, [0]); tag(274, 3, [orientation]);
    tag(277, 3, [1]); tag(278, 4, [height]); tag(279, 4, [width * height * 2]); tag(284, 3, [1]);
    tag(33421, 3, [2, 2]); tag(33422, 1, [0, 1, 1, 2]);
    tag(50706, 1, [1, 4, 0, 0]); tag(50707, 1, [1, 1, 0, 0]); tag(50708, 2, 'SoundKit Synthetic Bayer');
    tag(50714, 5, [64]); tag(50717, 4, [4095]);
    tag(50721, 10, [3.2406, -1.5372, -.4986, -.9689, 1.8758, .0415, .0557, -.204, 1.057]);
    tag(50728, 5, [1, 1, 1]); tag(50778, 3, [21]);
    tags.sort((a, b) => a.id - b.id);
    let offset = 8 + 2 + tags.length * 12 + 4;
    for (const entry of tags) if (entry.bytes.length > 4) { entry.offset = offset; offset += entry.bytes.length + entry.bytes.length % 2; }
    tags.find(entry => entry.id === 273).bytes.writeUInt32LE(offset);
    const result = Buffer.alloc(offset + width * height * 2);
    result.write('II'); result.writeUInt16LE(42, 2); result.writeUInt32LE(8, 4); result.writeUInt16LE(tags.length, 8);
    tags.forEach((entry, index) => {
        const pos = 10 + index * 12;
        result.writeUInt16LE(entry.id, pos); result.writeUInt16LE(entry.type, pos + 2); result.writeUInt32LE(entry.count, pos + 4);
        if (entry.offset) { result.writeUInt32LE(entry.offset, pos + 8); entry.bytes.copy(result, entry.offset); }
        else entry.bytes.copy(result, pos + 8);
    });
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const channel = y % 2 ? (x % 2 ? 2 : 1) : (x % 2 ? 1 : 0);
        // A neutral ramp on the left, red/green/blue panels on the right.
        const panel = Math.floor(y / height * 3), coloured = x > width * .7;
        const linear = coloured ? (channel === panel ? .6 : .06) : .03 + .65 * x / width;
        result.writeUInt16LE(Math.round(64 + linear * (4095 - 64)), offset + (y * width + x) * 2);
    }
    return result;
}
