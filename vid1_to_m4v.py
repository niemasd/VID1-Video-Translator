#!/usr/bin/env python3
'''
Rewrite Factor 5 VID1-video to MPEG-4 Visual ES

This is NOT a decoder. It assumes the VID1 macroblock/texture payload after the
Factor 5 picture header is already compatible with MPEG-4 Part 2 / ASP, and
only rewrites container packets + picture headers into a normal MPEG-4 Visual
elementary stream (.m4v).

Limitations:
  * S/GMC/sprite frames are rejected.
  * Per-frame VID1 extended quant matrices are rejected, because MPEG-4 Visual
    VOL matrices are intra/non-intra, not luma/chroma per-picture matrices.
  * If B-frame macroblock syntax differs from ISO/IEC 14496-2, ffmpeg will fail
    while decoding the resulting .m4v; that requires a real decoder or a deeper
    macroblock-level bitstream translator.
'''
# imports
import argparse
import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

# constants
TAG = lambda s: int.from_bytes(s.encode("ascii"), "big")
KNOWN_TAGS = {TAG("FRAM"), TAG("VIDD"), TAG("AUDD")}

# custom VID1 error class
class VID1Error(Exception):
    pass

# class to read binary data bit-by-bit from Most-Significant Bit (MSB)
class BitReaderMSB:
    # initialize
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    # read `n` bits from current bit position
    def read(self, n: int) -> int:
        if n < 0:
            raise ValueError("negative bit count")
        if self.bitpos + n > len(self.data) * 8:
            raise VID1Error("unexpected end of bitstream")
        v = 0
        for _ in range(n):
            b = self.data[self.bitpos >> 3]
            bit = (b >> (7 - (self.bitpos & 7))) & 1
            v = (v << 1) | bit
            self.bitpos += 1
        return v

    # byte-align current bit position
    def align_byte(self) -> None:
        self.bitpos = (self.bitpos + 7) & ~7

    # get current byte position
    @property
    def bytepos(self) -> int:
        if self.bitpos & 7:
            raise VID1Error("bit reader is not byte-aligned")
        return self.bitpos >> 3

# class to read binary data bit-by-bit from Least-Significant Bit (LSB)
class BitReaderLSB:
    # initialize
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    # read `n` bits from current bit position
    def read(self, n: int) -> int:
        if self.bitpos + n > len(self.data) * 8:
            raise VID1Error("unexpected end of little-endian bit header")
        v = 0
        for i in range(n):
            b = self.data[self.bitpos >> 3]
            bit = (b >> (self.bitpos & 7)) & 1
            v |= bit << i
            self.bitpos += 1
        return v

# class to write bits to binary output
class BitWriter:
    # initialize
    def __init__(self, out: BinaryIO):
        self.out = out
        self.cur = 0
        self.nbits = 0
        self.bytes_written = 0

    # write a single bit
    def write_bit(self, bit: int) -> None:
        self.cur = (self.cur << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.out.write(bytes([self.cur]))
            self.bytes_written += 1
            self.cur = 0
            self.nbits = 0

    # write `n` bits from integer `value`
    def write_bits(self, n: int, value: int) -> None:
        if n < 0:
            raise ValueError("negative bit count")
        if n and value >= (1 << n):
            raise ValueError(f"value {value} does not fit in {n} bits")
        for i in range(n - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    # flush
    def byte_align_zero(self) -> None:
        if self.nbits:
            self.cur <<= 8 - self.nbits
            self.out.write(bytes([self.cur]))
            self.bytes_written += 1
            self.cur = 0
            self.nbits = 0

    # write sequence of bytes as bits
    def write_bytes_as_bits(self, data: bytes) -> None:
        if not data:
            return
        if self.nbits == 0:
            self.out.write(data)
            self.bytes_written += len(data)
            return
        for b in data:
            self.write_bits(8, b)

    # write MPEG-4 start code
    def start_code(self, code: int) -> None:
        if not 0 <= code <= 0xFF:
            raise ValueError("start-code suffix must fit in one byte")
        self.byte_align_zero()
        self.out.write(b"\x00\x00\x01" + bytes([code]))
        self.bytes_written += 4

    # close output stream (currently just flush)
    def close(self) -> None:
        self.byte_align_zero()

# class to represent VID1 container info
@dataclass
class VID1Info:
    be: bool
    width: Optional[int]
    height: Optional[int]
    start_offset: int
    audio_codec: Optional[str]
    frame_count: Optional[int] = None
    fps_num: Optional[int] = None
    fps_den: Optional[int] = None

# class to represent a single frame
@dataclass
class Picture:
    frame_type: int  # 0 I, 1 P, 2 B, 3 S
    rounding: int
    intra_dc_vlc_thr_idx: int
    quant: int
    fcode_forward: int
    fcode_backward: int
    timecode: int
    payload: bytes
    ignored16: int
    extended_info_present: int
    sprite_info_present: int
    extended_quant: int

# read the next `n` bytes from a binary data stream
def read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b

# read the next uint16 from a binary data stream
def read_u16(f: BinaryIO, be: bool) -> int:
    return int.from_bytes(read_exact(f, 2), "big" if be else "little")

# read the next uint32 from a binary data stream
def read_u32(f: BinaryIO, be: bool) -> int:
    return int.from_bytes(read_exact(f, 4), "big" if be else "little")

# parse the VID1 container file header
def parse_vid1_file_header(f: BinaryIO) -> VID1Info:
    magic_bytes = read_exact(f, 4)
    magic_be = int.from_bytes(magic_bytes, "big")
    if magic_be == TAG("VID1"):
        be = True
    elif magic_be == int.from_bytes(b"1DIV", "big"):
        be = False
    else:
        raise VID1Error(f"not a VID1 file: first four bytes are {magic_bytes!r}")
    head_off = read_u32(f, be)
    f.seek(head_off)
    if read_u32(f, be) != TAG("HEAD"):
        raise VID1Error("HEAD chunk not found at header offset")
    start_offset = head_off + read_u32(f, be)
    width = height = None
    audio_codec = None
    frame_count = None
    fps_num = None
    fps_den = None
    off = head_off + 12
    f.seek(off)
    try:
        chunk = read_u32(f, be)
    except EOFError:
        raise VID1Error("truncated HEAD area")
    if chunk == TAG("VIDH"):
        f.seek(off + 4)
        vidh_size = read_u32(f, be)
        next_off = off + vidh_size
        f.seek(off + 8)
        f.seek(4, io.SEEK_CUR) # unknown/reserved according to librempeg demuxer
        width = read_u16(f, be)
        height = read_u16(f, be)
        remaining = next_off - f.tell()
        if remaining >= 4:
            frame_count = read_u32(f, be)
            remaining -= 4
        if remaining >= 4:
            _vidh_time_or_unknown = read_u32(f, be)
            remaining -= 4
        if remaining >= 4:
            fps_num = read_u32(f, be)
            remaining -= 4
        if remaining >= 2:
            den = read_u16(f, be)
            fps_den = den or None
            remaining -= 2
        off = next_off
    f.seek(off)
    try:
        chunk = read_u32(f, be)
    except EOFError:
        chunk = None
    if chunk == TAG("AUDH"):
        f.seek(off + 12)
        codec = read_u32(f, be)
        audio_codec = {
            TAG("PC16"): "PC16",
            TAG("XAPM"): "XAPM",
            TAG("APCM"): "APCM",
            TAG("VAUD"): "VAUD/Vorbis",
        }.get(codec, f"0x{codec:08x}")
    return VID1Info(
        be=be,
        width=width,
        height=height,
        start_offset=start_offset,
        audio_codec=audio_codec,
        frame_count=frame_count,
        fps_num=fps_num,
        fps_den=fps_den,
    )

# return a parsed VID1 variable packet header as a (header_len_bytes, packet_size) tuple
def parse_var_packet_header_at(f: BinaryIO, pos: int) -> tuple[int, int]:
    f.seek(pos)
    ibuf = f.read(4)
    if len(ibuf) < 4:
        raise EOFError
    br = BitReaderLSB(ibuf)
    size_bits = br.read(4)
    if size_bits > 30:
        raise VID1Error(f"unreasonable packet-header size_bits={size_bits} at 0x{pos:x}")
    size = br.read(size_bits + 1)
    if size_bits == 0 and size == 0 and ibuf[0] == 0x80:
        size = 1
    header_len = (br.bitpos + 7) // 8
    return header_len, size

# convert tag value to bytes
def on_disk_tag_bytes(tag_value: int, be: bool) -> bytes:
    b = tag_value.to_bytes(4, "big")
    return b if be else b[::-1]

# find the next known chunk
def find_next_known_chunk(f: BinaryIO, be: bool, start: int, max_scan: int = 1 << 20) -> Optional[int]:
    f.seek(start)
    data = f.read(max_scan)
    needles = [on_disk_tag_bytes(t, be) for t in KNOWN_TAGS]
    best = None
    for needle in needles:
        i = data.find(needle)
        if i >= 0:
            p = start + i
            best = p if best is None else min(best, p)
    return best

# iterate over video packets
def iter_video_packets(f: BinaryIO, info: VID1Info, *, resync_scan: bool = False, vidd_header_skip: int = 4) -> Iterator[bytes]:
    be = info.be
    f.seek(info.start_offset)
    while True:
        pos = f.tell()
        try:
            magic = read_u32(f, be)
        except EOFError:
            return
        if magic == TAG("FRAM"):
            f.seek(28, io.SEEK_CUR)
            try:
                magic = read_u32(f, be)
            except EOFError:
                return
        if magic == TAG("VIDD"):
            chunk_size = read_u32(f, be)
            if vidd_header_skip < 0:
                raise VID1Error("VIDD header skip must be non-negative")
            if chunk_size < 8 + vidd_header_skip:
                raise VID1Error(f"bad VIDD chunk size {chunk_size} at 0x{pos:x}")
            f.seek(vidd_header_skip, io.SEEK_CUR)
            pkt_size = chunk_size - 8 - vidd_header_skip
            pkt = read_exact(f, pkt_size)
            yield pkt
        elif magic == TAG("AUDD"):
            chunk_size = read_u32(f, be)
            if chunk_size < 16:
                raise VID1Error(f"bad AUDD chunk size {chunk_size} at 0x{pos:x}")
            f.seek(4, io.SEEK_CUR)
            _pkt_size = read_u32(f, be)
            f.seek(chunk_size - 16, io.SEEK_CUR)
        else:
            if magic == 0:
                tail = f.read()
                if not tail or all(b == 0 for b in tail):
                    return
                f.seek(pos + 4)
            try:
                header_len, pkt_size = parse_var_packet_header_at(f, pos)
                if pkt_size <= 0 or pkt_size > (1 << 28):
                    raise VID1Error("unreasonable bare audio packet size")
                f.seek(pos + header_len + pkt_size)
            except Exception:
                if not resync_scan:
                    raise VID1Error(
                        f"unknown chunk/tag 0x{magic:08x} at 0x{pos:x}; "
                        "try --resync-scan if this file has bare audio packets"
                    )
                nxt = find_next_known_chunk(f, be, pos + 1)
                if nxt is None:
                    return
                f.seek(nxt)

# parse a single frame from bytes
def parse_picture(pkt: bytes, frame_index: int) -> Picture:
    br = BitReaderMSB(pkt)
    ignored16 = br.read(16)
    frame_type = br.read(2)
    ext = br.read(1)
    sprite_info_present = 0
    extended_quant = 0
    if ext:
        sprite_info_present = br.read(1)
        if sprite_info_present:
            _num_sprites = br.read(2)
            _sprite_mv_resolution = br.read(2)
        extended_quant = br.read(1)
        if extended_quant:
            # Factor 5 says luma/chroma qmat here. MPEG-4 Visual VOL has intra/non-intra matrices instead, so do not silently mis-map it.
            luma_present = br.read(1)
            if luma_present:
                _luma_qmat = [br.read(8) for _ in range(64)]
            chroma_present = br.read(1)
            if chroma_present:
                _chroma_qmat = [br.read(8) for _ in range(64)]
        _ignored_a = br.read(1)
        _ignored_b = br.read(1)
    rounding = br.read(1)
    dc_thr = br.read(3)
    quant = br.read(5)
    fwd = 1
    back = 1
    if frame_type != 0:
        fwd = br.read(3)
    if frame_type == 2:
        back = br.read(3)
    timecode = br.read(32)
    if frame_type == 3:
        raise VID1Error(f"frame {frame_index}: S/GMC/sprite frame is not supported")
    if sprite_info_present:
        raise VID1Error(f"frame {frame_index}: sprite/GMC info is present; this prototype does not map it")
    if extended_quant:
        raise VID1Error(f"frame {frame_index}: extended luma/chroma quant matrices are present; not safely mappable to MPEG-4 VOL matrices")
    if quant == 0:
        raise VID1Error(f"frame {frame_index}: qscale 0 is invalid for MPEG-4 Visual")
    if frame_type != 0 and fwd == 0:
        raise VID1Error(f"frame {frame_index}: forward fcode 0 is invalid for MPEG-4 Visual")
    if frame_type == 2 and back == 0:
        raise VID1Error(f"frame {frame_index}: backward fcode 0 is invalid for MPEG-4 Visual")
    br.align_byte()
    return Picture(
        frame_type=frame_type,
        rounding=rounding,
        intra_dc_vlc_thr_idx=dc_thr,
        quant=quant,
        fcode_forward=fwd,
        fcode_backward=back,
        timecode=timecode,
        payload=pkt[br.bytepos:],
        ignored16=ignored16,
        extended_info_present=ext,
        sprite_info_present=sprite_info_present,
        extended_quant=extended_quant,
    )

# return bit length of time increment resolution
def time_increment_bits(time_res: int) -> int:
    if time_res <= 0 or time_res > 65535:
        raise ValueError("MPEG-4 vop_time_increment_resolution must be 1..65535")
    return max(1, (time_res - 1).bit_length())

# write visual headers
def write_visual_headers(bw: BitWriter, width: int, height: int, time_res: int) -> None:
    if not (1 <= width <= 8191 and 1 <= height <= 8191):
        raise VID1Error(f"MPEG-4 VOL width/height out of range: {width}x{height}")

    # Visual Object Sequence start, Advanced Simple Profile level 5.
    bw.start_code(0xB0)
    bw.write_bits(8, 0xF5)

    # Visual Object: is_visual_object_identifier=0, visual_object_type=video, video_signal_type=0.
    bw.start_code(0xB5)
    bw.write_bits(1, 0)
    bw.write_bits(4, 1)
    bw.write_bits(1, 0)
    bw.byte_align_zero()

    # Video Object start, object id 0.
    bw.start_code(0x00)

    # Video Object Layer start, layer id 0.
    bw.start_code(0x20)
    bw.write_bits(1, 0)       # random_accessible_vol
    bw.write_bits(8, 0x11)    # video_object_type_indication: Advanced Simple
    bw.write_bits(1, 1)       # is_object_layer_identifier
    bw.write_bits(4, 5)       # video_object_layer_verid
    bw.write_bits(3, 1)       # video_object_layer_priority
    bw.write_bits(4, 1)       # aspect_ratio_info: square pixels
    bw.write_bits(1, 0)       # vol_control_parameters absent
    bw.write_bits(2, 0)       # video_object_layer_shape: rectangular
    bw.write_bits(1, 1)       # marker_bit
    bw.write_bits(16, time_res)
    bw.write_bits(1, 1)       # marker_bit
    bw.write_bits(1, 0)       # fixed_vop_rate=false
    bw.write_bits(1, 1)       # marker before width
    bw.write_bits(13, width)
    bw.write_bits(1, 1)       # marker before height
    bw.write_bits(13, height)
    bw.write_bits(1, 1)       # marker after height
    bw.write_bits(1, 0)       # interlaced=false (progressive)
    bw.write_bits(1, 1)       # obmc_disable=true
    bw.write_bits(2, 0)       # vol_sprite_usage=not used; verid != 1 => 2 bits
    bw.write_bits(1, 0)       # not_8_bit=false => quant_precision=5, bpp=8
    bw.write_bits(1, 0)       # vol_quant_type=0, H.263 quantization
    bw.write_bits(1, 0)       # quarter_sample=false
    bw.write_bits(1, 1)       # complexity_estimation_disable=true
    bw.write_bits(1, 1)       # resync_marker_disable=true
    bw.write_bits(1, 0)       # data_partitioned=false
    bw.write_bits(1, 0)       # newpred_enable=false, verid != 1
    bw.write_bits(1, 0)       # reduced_resolution_vop_enable=false, verid != 1
    bw.write_bits(1, 0)       # scalability=false
    bw.byte_align_zero()

# choose an MPEG-4 VOP time resolution
def infer_time_res(info: VID1Info, fallback_fps: float) -> int:
    if info.fps_num and info.fps_den:
        fps = info.fps_num / info.fps_den
        ntsc_rates = [
            (24000 / 1001, 24000),
            (30000 / 1001, 30000),
            (60000 / 1001, 60000),
        ]
        for rate, time_res in ntsc_rates:
            if abs(fps - rate) < 0.02:
                return time_res
        if abs(fps - round(fps)) < 1e-6:
            return int(round(fps))
        return max(1, int(round(fps)))
    if abs(fallback_fps - round(fallback_fps)) > 1e-9:
        raise VID1Error("non-integer --fps needs explicit --time-res")
    return int(round(fallback_fps))

# class to represent time state
@dataclass
class TimeState:
    # member variables
    time_res: int
    use_vid1_timecode: bool
    first_timecode: Optional[int] = None
    frame_counter: int = 0
    time_base: int = 0
    last_time_base_for_b: int = 0

    # get the time of a given frame
    def picture_time(self, pic: Picture) -> int:
        if self.use_vid1_timecode:
            if self.first_timecode is None:
                self.first_timecode = pic.timecode
            t = pic.timecode - self.first_timecode
            if t < 0:
                t = self.frame_counter
        else:
            t = self.frame_counter
        self.frame_counter += 1
        return t

# write VOP header
def write_vop_header(bw: BitWriter, pic: Picture, ts: TimeState) -> None:
    t = ts.picture_time(pic)
    sec = t // ts.time_res
    inc = t % ts.time_res
    if pic.frame_type == 2: # B-VOP: time base is relative to last_time_base
        modulo = sec - ts.last_time_base_for_b
    else:
        modulo = sec - ts.time_base
        ts.last_time_base_for_b = ts.time_base
        ts.time_base = sec
    if modulo < 0:
        modulo = 0
    bw.start_code(0xB6)
    bw.write_bits(2, pic.frame_type)
    for _ in range(modulo):
        bw.write_bits(1, 1)
    bw.write_bits(1, 0)
    bw.write_bits(1, 1) # marker before vop_time_increment
    bw.write_bits(time_increment_bits(ts.time_res), inc)
    bw.write_bits(1, 1) # marker before vop_coded
    bw.write_bits(1, 1) # vop_coded
    if pic.frame_type == 1: # FFmpeg's MPEG-4 decoder reads this only for P, or S+GMC. We reject S/GMC.
        bw.write_bits(1, pic.rounding)
    bw.write_bits(3, pic.intra_dc_vlc_thr_idx)
    bw.write_bits(5, pic.quant)
    if pic.frame_type != 0:
        bw.write_bits(3, pic.fcode_forward)
    if pic.frame_type == 2:
        bw.write_bits(3, pic.fcode_backward)

# perform VID1 to M4V conversion
def convert(args: argparse.Namespace) -> None:
    with open(args.input, "rb") as f:
        info = parse_vid1_file_header(f)
        width = args.width or info.width
        height = args.height or info.height
        if width is None or height is None:
            raise VID1Error("width/height not found in VIDH; pass --width and --height")
        fps = args.fps
        time_res = args.time_res if args.time_res is not None else infer_time_res(info, fps)
        if time_res <= 0:
            raise VID1Error("time resolution must be positive")
        if args.verbose:
            rate = "unknown"
            if info.fps_num and info.fps_den:
                rate = f"{info.fps_num}/{info.fps_den}"
            print(
                f"VID1: endian={'BE' if info.be else 'LE'} "
                f"size={width}x{height} start=0x{info.start_offset:x} "
                f"frames(header)={info.frame_count} rate(header)={rate} "
                f"time_res={time_res} audio={info.audio_codec}",
                file=sys.stderr,
            )
        ts = TimeState(time_res=time_res, use_vid1_timecode=not args.synthetic_time)
        frames = 0
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        with open(args.output, "wb") as out:
            bw = BitWriter(out)
            write_visual_headers(bw, width, height, time_res)
            for pkt in iter_video_packets(
                f,
                info,
                resync_scan=args.resync_scan,
                vidd_header_skip=args.vidd_header_skip,
            ):
                pic = parse_picture(pkt, frames)
                counts[pic.frame_type] += 1
                write_vop_header(bw, pic, ts)
                bw.write_bytes_as_bits(pic.payload)
                bw.byte_align_zero()
                frames += 1
            bw.close()
    if args.verbose:
        print(
            f"wrote {frames} VOPs to {args.output} "
            f"(I={counts[0]} P={counts[1]} B={counts[2]})",
            file=sys.stderr,
        )
    if frames == 0:
        raise VID1Error("no video packets found")

# main program logic
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, help="Input Factor 5 VID1 File (.vid)")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output MPEG-4 Visual Elementary Stream (.m4v)")
    parser.add_argument('--fps', required=False, type=float, default=24.0, help="Fallback nominal frame rate if VIDH has no rate")
    parser.add_argument('--time-res', required=False, type=int, default=None, help="MPEG-4 vop_time_increment_resolution; defaults to VIDH-derived value when available")
    parser.add_argument('--synthetic-time', action='store_true', help="Ignore VID1 32-bit timecodes and number VOPs sequentially")
    parser.add_argument('--width', required=False, type=int, default=None, help="Override width if VIDH parsing fails")
    parser.add_argument('--height', required=False, type=int, default=None, help="Override height if VIDH parsing fails")
    parser.add_argument('--resync-scan', action='store_true', help="Scan forward for the next known VID1 chunk after an unknown tag")
    parser.add_argument('--vidd-header-skip', required=False, type=int, default=4, help="Bytes to skip after VIDD size before the VID1 picture header (librempeg currently uses 6)")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite Output File")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show Verbose Messages")
    args = parser.parse_args(argv)
    args.input = Path(args.input)
    if not args.input.is_file():
        raise ValueError(f"File not found: {args.input}")
    args.output = Path(args.output)
    if args.output.exists():
        if args.overwrite and args.output.is_file():
            args.output.unlink(missing_ok=True)
        else:
            raise ValueError(f"Output exists: {args.output}")
    try:
        convert(args)
    except Exception as e:
        args.output.unlink(missing_ok=True)
        raise e

# run program
if __name__ == "__main__":
    main()
