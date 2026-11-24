"""Update all visits' convolution histories and local K/V rings in two launches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_histories(DESCRIPTORS, ACCEPTED, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    descriptor = DESCRIPTORS + 3 * tl.program_id(1)
    width = tl.load(descriptor + 2).to(tl.int32)
    channel_row = tile * BLOCK + tl.arange(0, BLOCK)
    mask = channel_row < 3 * width
    source = tl.load(descriptor).to(tl.pointer_type(tl.bfloat16))
    target = tl.load(descriptor + 1).to(tl.pointer_type(tl.bfloat16))
    accepted = tl.load(ACCEPTED).to(tl.int32)
    values = tl.load(source + accepted * width + channel_row, mask, other=0)
    tl.store(target + channel_row, values, mask)


@triton.jit
def _commit_rings(DESCRIPTORS, ACCEPTED, POSITION, ROWS: tl.constexpr,
                  WINDOW: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
    descriptor = DESCRIPTORS + 2 * tl.program_id(1)
    source = tl.load(descriptor).to(tl.pointer_type(tl.float32))
    target = tl.load(descriptor + 1).to(tl.pointer_type(tl.float32))
    head = tl.program_id(2)
    row_lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = row_lane // HEAD_DIM
    lane = row_lane % HEAD_DIM
    accepted = tl.load(ACCEPTED).to(tl.int32)
    position = tl.load(POSITION)
    slot = ((position + row) % WINDOW).to(tl.int32)
    mask = (row < accepted) & (row < ROWS)
    values = tl.load(source + head * ROWS * HEAD_DIM + row_lane, mask, other=0)
    tl.store(target + (head * WINDOW + slot) * HEAD_DIM + lane, values, mask)


class CommitBuffers:
    """Hold the static buffers and device-side pointer tables for one graph shape."""

    def __init__(self, histories, rings):
        self.histories = tuple(histories)
        self.rings = tuple(rings)
        device = self.histories[0][0].device
        self.history_table = torch.tensor(
            [(src.data_ptr(), dst.data_ptr(), dst.shape[1]) for src, dst in self.histories],
            dtype=torch.int64,
            device=device,
        )
        self.history_tiles = triton.cdiv(3 * max(dst.shape[1] for _, dst in self.histories), 256)
        source, target = self.rings[0]
        self.heads, self.rows, self.head_dim = source.shape
        self.window = target.shape[1]
        self.ring_table = torch.tensor(
            [(src.data_ptr(), dst.data_ptr()) for src, dst in self.rings],
            dtype=torch.int64,
            device=device,
        )

    def commit(self, accepted, position):
        _commit_histories[(self.history_tiles, len(self.histories))](
            self.history_table, accepted, 256, num_warps=4,
        )
        tiles = triton.cdiv(self.rows * self.head_dim, 128)
        _commit_rings[(tiles, len(self.rings), self.heads)](
            self.ring_table, accepted, position, self.rows, self.window,
            self.head_dim, 128, num_warps=4,
        )
