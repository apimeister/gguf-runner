use crate::engine::distributed::protocol::{
    DISTRIBUTED_PROTOCOL_MAGIC, DISTRIBUTED_PROTOCOL_VERSION, FrameKind,
};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

const HEADER_LEN: usize = 20;

#[derive(Clone, Debug)]
pub(crate) struct TransportMessage {
    pub(crate) kind: FrameKind,
    pub(crate) request_id: u64,
    pub(crate) payload: Vec<u8>,
}

pub(crate) struct FramedConnection {
    stream: TcpStream,
}

impl FramedConnection {
    pub(crate) fn connect(address: &str, timeout: Duration) -> Result<Self, String> {
        let stream = TcpStream::connect(address)
            .map_err(|e| format!("failed to connect to '{address}': {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("failed to set TCP_NODELAY for '{address}': {e}"))?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("failed to set read timeout for '{address}': {e}"))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|e| format!("failed to set write timeout for '{address}': {e}"))?;
        Ok(Self { stream })
    }

    pub(crate) fn from_stream(stream: TcpStream, timeout: Duration) -> Result<Self, String> {
        stream
            .set_nodelay(true)
            .map_err(|e| format!("failed to set TCP_NODELAY: {e}"))?;
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("failed to set read timeout: {e}"))?;
        stream
            .set_write_timeout(Some(timeout))
            .map_err(|e| format!("failed to set write timeout: {e}"))?;
        Ok(Self { stream })
    }

    pub(crate) fn try_clone(&self, timeout: Duration) -> Result<Self, String> {
        let stream = self
            .stream
            .try_clone()
            .map_err(|e| format!("failed to clone distributed stream: {e}"))?;
        Self::from_stream(stream, timeout)
    }

    pub(crate) fn send_message(
        &mut self,
        kind: FrameKind,
        request_id: u64,
        payload: &[u8],
    ) -> Result<(), String> {
        let payload_len_u32: u32 = payload
            .len()
            .try_into()
            .map_err(|_| "payload too large for distributed frame".to_string())?;
        let mut header = [0u8; HEADER_LEN];
        header[0..4].copy_from_slice(&DISTRIBUTED_PROTOCOL_MAGIC.to_le_bytes());
        header[4..6].copy_from_slice(&DISTRIBUTED_PROTOCOL_VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&(kind as u16).to_le_bytes());
        header[8..16].copy_from_slice(&request_id.to_le_bytes());
        header[16..20].copy_from_slice(&payload_len_u32.to_le_bytes());
        self.stream
            .write_all(&header)
            .map_err(|e| format!("failed to write distributed frame header: {e}"))?;
        self.stream
            .write_all(payload)
            .map_err(|e| format!("failed to write distributed frame payload: {e}"))?;
        self.stream
            .flush()
            .map_err(|e| format!("failed to flush distributed frame: {e}"))?;
        Ok(())
    }

    pub(crate) fn set_timeout(&mut self, timeout: Duration) -> Result<(), String> {
        self.stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("failed to set read timeout: {e}"))?;
        self.stream
            .set_write_timeout(Some(timeout))
            .map_err(|e| format!("failed to set write timeout: {e}"))?;
        Ok(())
    }

    pub(crate) fn send_raw_slices(
        &mut self,
        mapped: &[u8],
        ranges: &[(usize, usize)],
    ) -> Result<(), String> {
        let total: u64 = ranges.iter().map(|(_, l)| *l as u64).sum();
        self.stream
            .write_all(&total.to_le_bytes())
            .map_err(|e| format!("failed to write shard length: {e}"))?;
        for &(offset, len) in ranges {
            self.stream
                .write_all(&mapped[offset..offset + len])
                .map_err(|e| format!("failed to write shard data: {e}"))?;
        }
        self.stream
            .flush()
            .map_err(|e| format!("failed to flush shard: {e}"))?;
        Ok(())
    }

    pub(crate) fn recv_raw_stream_to_file(
        &mut self,
        file: &mut std::fs::File,
    ) -> Result<u64, String> {
        let mut len_buf = [0u8; 8];
        self.stream
            .read_exact(&mut len_buf)
            .map_err(|e| format!("failed to read shard length: {e}"))?;
        let total = u64::from_le_bytes(len_buf);
        const CHUNK: usize = 8 * 1024 * 1024;
        let mut buf = vec![0u8; CHUNK];
        let mut remaining = total as usize;
        while remaining > 0 {
            let to_read = remaining.min(CHUNK);
            self.stream
                .read_exact(&mut buf[..to_read])
                .map_err(|e| format!("failed to read shard data: {e}"))?;
            file.write_all(&buf[..to_read])
                .map_err(|e| format!("failed to write shard to file: {e}"))?;
            remaining -= to_read;
        }
        file.flush()
            .map_err(|e| format!("failed to flush shard file: {e}"))?;
        Ok(total)
    }

    pub(crate) fn recv_message(&mut self) -> Result<TransportMessage, String> {
        let mut header = [0u8; HEADER_LEN];
        self.stream
            .read_exact(&mut header)
            .map_err(|e| format!("failed to read distributed frame header: {e}"))?;
        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != DISTRIBUTED_PROTOCOL_MAGIC {
            return Err(format!(
                "invalid distributed frame magic: expected 0x{DISTRIBUTED_PROTOCOL_MAGIC:08X}, got 0x{magic:08X}"
            ));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != DISTRIBUTED_PROTOCOL_VERSION {
            return Err(format!(
                "unsupported distributed protocol version {version}, expected {DISTRIBUTED_PROTOCOL_VERSION}"
            ));
        }
        let kind = FrameKind::from_u16(u16::from_le_bytes([header[6], header[7]]))?;
        let request_id = u64::from_le_bytes([
            header[8], header[9], header[10], header[11], header[12], header[13], header[14],
            header[15],
        ]);
        let payload_len =
            u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize;
        let mut payload = vec![0u8; payload_len];
        self.stream
            .read_exact(&mut payload)
            .map_err(|e| format!("failed to read distributed frame payload: {e}"))?;
        Ok(TransportMessage {
            kind,
            request_id,
            payload,
        })
    }
}
