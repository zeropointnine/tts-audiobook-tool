"""Perth implicit watermarker leaks CUDA memory on every call.

Chatterbox watermarks every generated clip with
`perth.PerthImplicitWatermarker.apply_watermark()`, which runs with autograd
enabled. `perth_net.encoder` then leaks a fixed amount of CUDA memory per call
that nothing can reclaim: `gc.collect()` finds 0 uncollectable objects, no
Python frame or module global references the tensors, and
`torch.cuda.empty_cache()` frees none of it because the blocks are live. Only
process exit releases it, which is why `Options > Unload models` looks like it
helps.

The trigger is in `perth/perth_net/perth_net_implicit/model/encoder.py`:

    sub_mag = magspec[:, : self.subband]   # view fed into the conv stack
    res = self.layers(sub_mag) * mask
    magspec[:, : self.subband] += res      # in-place add to the same tensor

The conv nodes save the view (whose `_base` is `magspec`) and the `CopySlices`
node created by the in-place add saves the previous version of `magspec`. That
closes a reference cycle made only of C++ autograd nodes, which Python's cyclic
GC never sees, so the whole encoder graph survives forever.

Measured on an RTX 3080 Ti, per call, 1.0 s of audio (magspec 1025x76):

    A. encoder.layers() only, no clone/mask/in-place      0.00 MiB    0 retained
    B. clone + convs + out-of-place add                   0.00 MiB    0 retained
    C. clone + convs on VIEW + in-place slice add         1.19 MiB  560 retained
    D. same as C, locals explicitly deleted               1.19 MiB  560 retained
    E. pn.encoder() full, locals explicitly deleted       1.19 MiB  560 retained

(560 retained tensors over 40 calls = 14 autograd tensors per call, never
released.)

So the leak is ~1.19 MiB per second of watermarked audio, and it scales with
clip length. Wrapping the call in `torch.no_grad()` or `torch.inference_mode()`
makes it exactly zero, and that is enough on its own: patching only
`apply_watermark` takes the real Chatterbox Turbo path from 6.18 MiB per
generation to 0.00 MiB.

Note that `ChatterboxModel._use_gpu_watermarker()` — since removed — moved this
leak from host RAM to VRAM rather than eliminating it. On a 12 GB card that was
strictly worse: ~6.3 GB of headroom is about 88 minutes of generated audio
before the next allocation fails. The fix now in `ChatterboxModel.generate()` is
`torch.inference_mode()`, which zeroes the leak on either device.

Run: ./venv-cb/bin/python -m testx.perth_watermark_vram
"""

import gc

import numpy as np
import perth
import torch

SR = 24000
SECONDS = 1.0
N = 40


def count_live_grad_tensors() -> int:
    n = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.grad_fn is not None:
                n += 1
        except ReferenceError:
            continue
    return n


def measure(label: str, fn, n: int = N) -> None:
    gc.collect()
    before = torch.cuda.memory_allocated()
    tensors_before = count_live_grad_tensors()
    for _ in range(n):
        fn()
    gc.collect()
    delta = (torch.cuda.memory_allocated() - before) / 2**20
    retained = count_live_grad_tensors() - tensors_before
    print(f"{label:52s} +{delta:7.1f} MiB ({delta/n:5.2f}/call)"
          f"  retained grad tensors={retained}", flush=True)


def main() -> None:
    torch.manual_seed(0)
    signal = (np.random.randn(int(SR * SECONDS)) * 0.1).astype(np.float32)

    watermarker = perth.PerthImplicitWatermarker(device="cuda")
    net = watermarker.perth_net
    mag = net.ap.signal_to_magphase(torch.from_numpy(signal).to(net.device))[0][None]
    mag = mag.to(net.device)
    subband = net.subband
    print(f"magspec {tuple(mag.shape)}  subband={subband}  "
          f"hidden={net.hp.hidden_size}  hop={net.hp.hop_size}\n")

    def full_call() -> None:
        watermarker.apply_watermark(signal, SR)

    def full_call_no_grad() -> None:
        with torch.no_grad():
            watermarker.apply_watermark(signal, SR)

    def convs_only() -> None:
        net.encoder.layers(mag[:, :subband].contiguous())

    def out_of_place() -> None:
        m = mag.clone()
        s = m.sum(dim=1)
        mask = (s > s.max(dim=1).values[:, None] * 0.05)[:, None].float()
        res = net.encoder.layers(m[:, :subband]) * mask
        out = m.clone()
        out[:, :subband] = out[:, :subband] + res

    def in_place_on_view() -> None:
        m = mag.clone()
        s = m.sum(dim=1)
        mask = (s > s.max(dim=1).values[:, None] * 0.05)[:, None].float()
        res = net.encoder.layers(m[:, :subband]) * mask
        m[:, :subband] += res

    measure("1. apply_watermark (as Chatterbox calls it)", full_call)
    measure("2. apply_watermark under torch.no_grad()", full_call_no_grad)
    measure("3. encoder.layers() only", convs_only)
    measure("4. clone + convs + out-of-place add", out_of_place)
    measure("5. clone + convs on VIEW + in-place slice add", in_place_on_view)


if __name__ == "__main__":
    main()
