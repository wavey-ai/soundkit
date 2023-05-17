use soundkit::{
    wav_to_opus_stream,
};

use structopt::StructOpt;
use std::path::PathBuf;
use std::io::{self, Read, Write, Error};
use std::fs::File;

#[derive(Debug, StructOpt)]
#[structopt(name = "soundkit", about = "An audio encoding utility.")]
struct Command {
    /// Commands
    #[structopt(subcommand)]
    cmd: Option<SubCommand>,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    /// Sampling rate
    #[structopt(short, long, default_value = "48000")]
    sampling_rate: u32,

    /// Bits per sample
    #[structopt(short, long, default_value = "16")]
    bits_per_sample: u8,

    /// Channel count
    #[structopt(short, long, default_value = "2")]
    channel_count: u8,

    /// Frame size
    #[structopt(short, long, default_value = "480")]
    frame_size: u16,
}

#[derive(Debug, StructOpt)]
enum SubCommand {
    /// Encode raw audio to Opus
    Encode,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Command::from_args();
    match args.cmd {
        Some(SubCommand::Encode) => {
            let mut buffer = Vec::new();

            match args.output {
                Some(ref path) => {
                    let mut file = std::fs::File::open(path)?;
                    file.read_to_end(&mut buffer)?;
                }
                None => {
                    io::stdin().read_to_end(&mut buffer)?;
                }
            }

            let encoded = wav_to_opus_stream(buffer.as_slice())?;

            match args.output {
                Some(path) => {
                    let mut file = std::fs::File::create(path)?;
                    file.write_all(&encoded)?;
                }
                None => {
                    io::stdout().write_all(&encoded)?;
                }
            }
        }
        None => {
            // Handle case where no subcommand was provided
            eprintln!("No command specified");
            std::process::exit(1);
        }
    }

    Ok(())
}

