// SPDX-License-Identifier: LGPL-2.1-or-later

use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut coding_unit = Vec::new();
    std::io::stdin().read_to_end(&mut coding_unit)?;
    let frame = soundkit_dnx::decode_frame(&coding_unit)?;
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for plane in frame.planes {
        if frame.bit_depth <= 8 {
            for sample in plane.samples {
                output.write_all(&[sample as u8])?;
            }
        } else {
            for sample in plane.samples {
                output.write_all(&sample.to_le_bytes())?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
