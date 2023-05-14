use std::net::SocketAddr;
use std::io::{self, Read, Write};
use std::time::Duration;

use srt_tokio::SrtSocket;
use srt_tokio::srt::{self, SrtCongestionCtrl};

pub struct SRTTransceiver {
    host: String,
    stream_id: String,
    receive: bool,
    ss: SrtSocket,
    bytes_sent: usize,
    mbps: f32,
}

impl SRTTransceiver {
    pub async fn new() -> io::Result<Self> {
        let host = "139.162.212.106".to_owned();
        let stream_id = Self::generate_stream_id();

        Ok(Self {
            host,
            stream_id,
            receive: false,
            ss: SrtSocket::new(srt::SockType::Datagram).await?,
            bytes_sent: 0,
            mbps: 0.0,
        })
    }

    fn generate_stream_id() -> String {
        let uuid = uuid::Uuid::new_v4();
        format!("{}-{}", uuid.to_simple_ref(), "0")
    }

    pub fn setup_to_send(&mut self) {
        self.stream_id = Self::generate_stream_id();
    }

    pub fn setup_to_receive(&mut self, stream_id: String) {
        self.stream_id = stream_id;
        self.receive = true;
    }

    fn get_stats(&self) -> io::Result<srt::TraceBStats> {
        let mut stats = srt::TraceBStats::default();
        self.ss.bistats(&mut stats)?;

        Ok(stats)
    }

    pub async fn connect_srt(&mut self) -> io::Result<()> {
        let sa: SocketAddr = format!("{}:1337", self.host).parse()?;

        let mut options = srt::SrtSocketOpts::default();
        options.set_send_latency(1)?;
        options.set_recv_latency(0)?;
        options.set_tsbpd_mode(true)?;
        options.set_nak_report(true)?;
        options.set_retransmit_algorithm(srt::RetransmitAlgo::Off)?;
        options.set_payload_size(srt::SrtLiveMaxPlSize as u32)?;
        options.set_congestion_ctrl(SrtCongestionCtrl::Live)?;

        self.ss.set_sock_opt(&srt::SockOpt::StreamId(self.stream_id.clone()))?;
        self.ss.set_sock_opts(&options)?;
        self.ss.connect(&sa).await?;

        Ok(())
    }

    pub async fn disconnect_srt(&mut self) -> io::Result<()> {
        self.ss.close().await?;

        Ok(())
    }

    pub async fn send(&mut self, data: &[u8]) -> io::Result<usize> {
        let nb = self.ss.send(data).await?;
        self.bytes_sent += nb;

        Ok(nb)
    }

    pub async fn receive(&mut self, stream_id: String) -> Result<(), Box<dyn Error>> {
        let listen_addr = format!("{}:{}", self.listen_address, self.listen_port);
        let socket = SrtSocketBuilder::new(SrtSocketType::Passive)
            .local_port(self.listen_port)
            .latency(self.latency)
            .build()
            .await?;
        socket.listen().await?;

        let mut buffer = vec![0u8; self.buffer_size];
        loop {
            let (size, _sockaddr) = socket.recv_from(&mut buffer).await?;
            let buffer = &buffer[..size];
            let packet = SrtPacket::parse(buffer)?;
            match packet {
                SrtPacket::Data(data_packet) => {
                    if data_packet.stream_id != stream_id {
                        continue;
                    }
                    let payload = &data_packet.payload;
                    let pcm_samples = decode_audio_packet(payload)?;
                    self.play_samples(&pcm_samples).await;
                }
                SrtPacket::Control(_control_packet) => {}
                SrtPacket::Ack(_ack_packet) => {}
                SrtPacket::Nak(_nak_packet) => {}
                SrtPacket::Shutdown(_shutdown_packet) => {
                    break;
                }
            }
        }
        Ok(())
    }
}

