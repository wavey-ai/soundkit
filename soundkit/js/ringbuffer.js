
function ringbuffer_from_data(data, dataType, frame_size) {
  const data_offset = 6 * 4;
  const el_bytes = dataType.BYTES_PER_ELEMENT;
  const sab = new SharedArrayBuffer(data.byteLength + data_offset);
  const sabView = new Uint8Array(sab);
  sabView.set(data, data_offset);
  const n = (data.byteLength / el_bytes) / frame_size;
  const w_ptr_b = new Uint32Array(sab, 12, 1);
  const in_b = new Uint32Array(sab, 0, 1);
  Atomics.store(w_ptr_b, 0, n);
  Atomics.store(in_b, 0, n);
  return ringbuffer(sab, frame_size, n, dataType)
}

function sharedbuffer_growable(frame_size, max_frames, dataType) {
  const data_offset = 6 * 4; // Assuming 4 bytes per Uint32Array element
  const el_bytes = dataType.BYTES_PER_ELEMENT;
  const initialLength = (frame_size * el_bytes) + data_offset;
  const maxByteLength = (frame_size * el_bytes * max_frames) + data_offset;
  const sab = new SharedArrayBuffer(
    initialLength,
    { maxByteLength },
  );

  return sab;
}

function sharedbuffer(frame_size, max_frames, dataType) {
  const data_offset = 6 * 4; // Assuming 4 bytes per Uint32Array element
  const el_bytes = dataType.BYTES_PER_ELEMENT;

  const sab = new SharedArrayBuffer((frame_size * el_bytes * max_frames) + data_offset);

  return sab;
}

function ringbuffer(sab, frame_size, max_frames, dataType) {
  const data_offset = 6 * 4; // Assuming 4 bytes per Uint32Array element
  const in_b = new Uint32Array(sab, 0, 1);
  const out_b = new Uint32Array(sab, 4, 1);
  const dropped_b = new Uint32Array(sab, 8, 1);
  const w_ptr_b = new Uint32Array(sab, 12, 1);
  const r_ptr_b = new Uint32Array(sab, 16, 1);
  const wrap_flag_b = new Uint32Array(sab, 20, 1);
  const el_bytes = dataType.BYTES_PER_ELEMENT;
  let data_b = new dataType(sab, data_offset, ((sab.byteLength - data_offset) / el_bytes));
  const frame_len = frame_size * el_bytes;
  const max_len = (frame_len * max_frames) + data_offset;
  const in_count = () => Atomics.load(in_b, 0);
  const out_count = () => Atomics.load(out_b, 0);
  const dropped_count = () => Atomics.load(dropped_b, 0);
  const w_ptr = () => Atomics.load(w_ptr_b, 0);
  const r_ptr = () => Atomics.load(r_ptr_b, 0);
  const wrap_flag = () => Atomics.load(wrap_flag_b, 0) === 1;

  const current_offset = (count_func) => {
    return (count_func() * frame_size);
  };

  const wrapping_add = (count_func) => {
    const i = count_func();
    return i == max_frames - 1 ? 0 : i + 1;
  };

  const count = () => {
    return in_count() - out_count();
  }

  const push = (frame) => {
    if (sab.growable) {
      if (sab.byteLength - data_offset == current_offset(w_ptr) * el_bytes) {
        if (sab.byteLength - data_offset == max_len) {
        } else {
          sab.grow(sab.byteLength + frame_len);
          data_b = new dataType(sab, data_offset, ((sab.byteLength - data_offset) / el_bytes));
        }
      }
    }
    const offset = current_offset(w_ptr);
    for (let i = 0; i < frame_size; i++) {
      data_b[offset + i] = frame[i];
    }

    if (offset === 0) {
      Atomics.store(wrap_flag_b, 0, 1);
    }

    if (wrap_flag() && current_offset(r_ptr) === offset) {
      Atomics.store(wrap_flag_b, 0, 0);
      const dropped = dropped_count() + (in_count() - out_count());
      Atomics.store(dropped_b, 0, dropped);
      Atomics.store(out_b, 0, in_count());
    }

    Atomics.add(in_b, 0, 1);
    Atomics.store(w_ptr_b, 0, wrapping_add(w_ptr));

    return true;
  };

  const pop = () => {
    if (in_count() - out_count() === 0) {
      return;
    }

    data_b = new dataType(sab, data_offset, ((sab.byteLength - data_offset) / el_bytes));
    const res = [];
    let offset = current_offset(r_ptr);
    for (let i = 0; i < frame_size; i++) {
      res.push(data_b[offset + i]);
    }
    Atomics.add(out_b, 0, 1);
    Atomics.store(r_ptr_b, 0, wrapping_add(r_ptr));

    return new dataType(res);
  };

  return {
    sab,
    push,
    pop,
    dropped_count,
    count,
  };
}

if (typeof module !== 'undefined') {
  module.exports = {
    ringbuffer,
    ringbuffer_from_data,
    sharedbuffer,
    sharedbuffer_growable,
  }
}
