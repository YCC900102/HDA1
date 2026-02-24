# trans_superposition_ldpc_ms.py
# ------------------------------------------------------------
# Two-branch DIGITAL superposition + SIC + Multi-step DDPM residual denoise (RX)
#
# Output ONLY (as you requested):
#   RX_base_x_hat.png     (upper-branch final)
#   RX_residual_r0_hat.png (lower-branch final, after multi-step reverse)
#   RX_final_x.png        (final combined)
# ------------------------------------------------------------

import os, io
import numpy as np
import torch
from PIL import Image

from modules_CDiff import UNet

# --- Require TF + Sionna (same as infer.py) ---
import tensorflow as tf
import sionna as sn
import sionna.fec.ldpc.encoding as ldpc_enc
import sionna.fec.ldpc.decoding as ldpc_dec


# =========================
# Image / numeric helpers
# =========================
def load_rgb_no_resize(path, img_size: int):
    img = Image.open(path).convert("RGB")
    if img.size != (img_size, img_size):
        raise ValueError(f"Input image must be {img_size}x{img_size}, but got {img.size}. "
                         f"Please resize the image beforehand.")
    return img

def pil_to_float01(pil_img):
    return np.asarray(pil_img).astype(np.float32) / 255.0

def float01_to_pil(x01):
    x01 = np.clip(x01, 0.0, 1.0)
    u8 = (x01 * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8)

def to_m11(x01):
    return x01 * 2.0 - 1.0

def to_01(xm11):
    return (xm11 + 1.0) * 0.5

def save_img_m11(xm11, out_path):
    float01_to_pil(to_01(xm11)).save(out_path)

def jpeg_encode_bytes(pil_img, quality=10):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality),
                 subsampling="4:2:0", optimize=True)
    return buf.getvalue()

def jpeg_decode_bytes(jpeg_bytes):
    buf = io.BytesIO(jpeg_bytes)
    return Image.open(buf).convert("RGB")


# =========================
# Diffusion schedules (match your multi-step script idea)
# =========================
def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.0095):
    # aligned to your trans_ms.py comment
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

def compute_schedules(betas: np.ndarray):
    alphas = (1.0 - betas).astype(np.float32)
    alpha_hat = np.cumprod(alphas).astype(np.float32)

    alpha_hat_prev = np.empty_like(alpha_hat)
    alpha_hat_prev[0] = 1.0
    alpha_hat_prev[1:] = alpha_hat[:-1]

    beta_tilde = betas * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)
    beta_tilde = np.clip(beta_tilde, 1e-20, 1.0).astype(np.float32)
    return alphas, alpha_hat, beta_tilde

def forward_diffuse(x0, t, alpha_hat, rng):
    """x_t = sqrt(alpha_hat[t]) * x0 + sqrt(1-alpha_hat[t]) * eps, t in [1..T]"""
    assert 1 <= t <= len(alpha_hat)
    ah = float(alpha_hat[t - 1])
    eps = rng.normal(0.0, 1.0, size=x0.shape).astype(np.float32)
    xt = np.sqrt(ah).astype(np.float32) * x0 + np.sqrt(1.0 - ah).astype(np.float32) * eps
    return xt.astype(np.float32)

@torch.no_grad()
def reverse_residual_multistep_ddpm(
    model,
    r_t_rx: np.ndarray,   # (H,W,3) in [-1,1]
    x_cond: np.ndarray,   # (H,W,3) in [-1,1]  (IMPORTANT: use RX base x_hat)
    t_start: int,
    betas: np.ndarray,
    alphas: np.ndarray,
    alpha_hat: np.ndarray,
    beta_tilde: np.ndarray,
    device: str,
    rng: np.random.Generator,
):
    """
    DDPM reverse on residual (epsilon prediction):
      mu_t = 1/sqrt(alpha_t) * (r_t - (1-alpha_t)/sqrt(1-\bar{alpha}_t) * eps_hat)
      r_{t-1} = mu_t + sqrt(beta_tilde_t) * z, z~N(0,I) for t>1; t=1 uses mu only.
    """
    r = torch.from_numpy(r_t_rx).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    c = torch.from_numpy(x_cond).permute(2, 0, 1).unsqueeze(0).to(device)

    H, W = r.shape[2], r.shape[3]

    for t in range(t_start, 0, -1):
        beta_t = float(betas[t - 1])
        alpha_t = float(alphas[t - 1])
        ah_t = float(alpha_hat[t - 1])
        bt_tilde = float(beta_tilde[t - 1])

        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        eps_hat = model(r, t_tensor, c)

        alpha_t_t = torch.tensor(alpha_t, device=device, dtype=torch.float32)
        ah_t_t = torch.tensor(ah_t, device=device, dtype=torch.float32)

        coef = (1.0 - alpha_t_t) / torch.sqrt(1.0 - ah_t_t)
        mu = (r - coef * eps_hat) / torch.sqrt(alpha_t_t)

        if t > 1:
            z = rng.normal(0.0, 1.0, size=(1, 3, H, W)).astype(np.float32)
            z = torch.from_numpy(z).to(device)
            sigma = torch.tensor(np.sqrt(bt_tilde), device=device, dtype=torch.float32)
            r = mu + sigma * z
        else:
            r = mu

    r0_hat = r[0].permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
    return r0_hat


# =========================
# Quantization for rt (TX/RX)  (same as before, needed for digital residual)
# =========================
def quantize_symmetric_uniform(x: np.ndarray, bits: int = 8, clip_k_sigma: float = 3.0):
    assert bits in (4, 6, 8)
    x = x.astype(np.float32)
    sigma = float(np.std(x) + 1e-8)
    A = float(clip_k_sigma * sigma)
    x_clip = np.clip(x, -A, A)

    L = 2 ** bits
    delta = (2.0 * A) / (L - 1)

    q = np.round((x_clip + A) / delta).astype(np.int32)
    q = np.clip(q, 0, L - 1).astype(np.uint8)

    meta = {"bits": bits, "A": A, "delta": float(delta), "shape": x.shape}
    return q, meta

def dequantize_symmetric_uniform(q_uint: np.ndarray, meta: dict):
    A = float(meta["A"])
    delta = float(meta["delta"])
    shape = tuple(meta["shape"])
    q = q_uint.astype(np.int32).reshape(-1)
    x_hat = (q.astype(np.float32) * delta - A).astype(np.float32)
    return x_hat.reshape(shape)

def u8_to_bits(u8: np.ndarray) -> np.ndarray:
    return np.unpackbits(u8.reshape(-1).astype(np.uint8)).astype(np.int32)

def bits_to_u8(bits: np.ndarray, num_u8: int) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8))
    if packed.size < num_u8:
        packed = np.pad(packed, (0, num_u8 - packed.size), mode="constant")
    return packed[:num_u8].astype(np.uint8)

def bytes_to_bits_u8(data: bytes) -> np.ndarray:
    u8 = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(u8)  # 0/1
    return bits.astype(np.int32)

def bits_to_bytes_u8(bits: np.ndarray) -> bytes:
    bits_u8 = bits.astype(np.uint8)
    packed = np.packbits(bits_u8)
    return packed.tobytes()


# =========================
# Sionna modem
# =========================
class SionnaModem:
    def __init__(self, mod_type: str):
        self.mod_type = mod_type.lower()
        if self.mod_type == "bpsk":
            self.num_bits_per_symbol = 1
            constellation_type = "pam"
        elif self.mod_type == "qpsk":
            self.num_bits_per_symbol = 2
            constellation_type = "qam"
        elif self.mod_type == "16qam":
            self.num_bits_per_symbol = 4
            constellation_type = "qam"
        else:
            raise ValueError(f"Unsupported mod_type: {mod_type}")

        self.mapper = sn.mapping.Mapper(
            constellation_type=constellation_type,
            num_bits_per_symbol=self.num_bits_per_symbol
        )
        self.demapper = sn.mapping.Demapper(
            "app",
            constellation_type=constellation_type,
            num_bits_per_symbol=self.num_bits_per_symbol
        )

    def modulate(self, bits_tf: tf.Tensor) -> tf.Tensor:
        return self.mapper(bits_tf)

    def demodulate_llr(self, y_mod: tf.Tensor, noise_var: tf.Tensor) -> tf.Tensor:
        return self.demapper([y_mod, noise_var])


# =========================
# LDPC + Modem pipeline
# =========================
def ldpc_modulate(bits01: np.ndarray, rate: str, modem: SionnaModem):
    bits01 = bits01.astype(np.int32).reshape(-1)
    orig_len = int(bits01.size)

    if rate == "1/2":
        k, n = 336, 672
    elif rate == "3/4":
        k, n = 504, 672
    else:
        raise ValueError("rate must be '1/2' or '3/4'")

    num_blocks = int(np.ceil(orig_len / k))
    pad_len = num_blocks * k - orig_len
    if pad_len > 0:
        bits01 = np.concatenate([bits01, np.zeros(pad_len, dtype=np.int32)], axis=0)

    bits_blocks = bits01.reshape(num_blocks, k)

    encoder = ldpc_enc.LDPC5GEncoder(k=k, n=n)
    decoder = ldpc_dec.LDPC5GDecoder(encoder, num_iter=10)

    b_tf = tf.convert_to_tensor(bits_blocks, dtype=tf.float32)  # [B,k]
    b_coded = encoder(b_tf)                                     # [B,n]
    b_coded_i = tf.cast(b_coded, tf.int32)
    b_stream = tf.reshape(b_coded_i, [1, num_blocks * n])       # [1, B*n]

    x_mod = modem.modulate(b_stream)                            # [1, Ns]
    p = tf.reduce_mean(tf.abs(x_mod) ** 2)
    x_mod = x_mod / tf.cast(tf.sqrt(p + 1e-8), tf.complex64)

    side = {"k": k, "n": n, "num_blocks": num_blocks, "orig_len": orig_len, "decoder": decoder}
    return x_mod, side

def demod_ldpc_decode(y_mod: tf.Tensor, noise_var: float, modem: SionnaModem, side: dict):
    decoder = side["decoder"]
    n = side["n"]
    num_blocks = side["num_blocks"]
    orig_len = side["orig_len"]

    nv = tf.constant(noise_var, dtype=tf.float32)
    llr = modem.demodulate_llr(y_mod, nv)

    # KEY FIX: flatten -> crop -> reshape
    llr = tf.reshape(llr, [1, -1])
    expected = num_blocks * n
    llr = llr[:, :expected]
    llr = tf.reshape(llr, [num_blocks, n])

    b_hat = decoder(llr)                       # [B,k] float 0/1
    b_hat = tf.cast(tf.round(b_hat), tf.int32).numpy().reshape(-1)
    b_hat = b_hat[:orig_len]
    return b_hat.astype(np.int32)

def pad_to(x, N):
    cur = int(x.shape[1])
    if cur == N:
        return x
    pad = tf.zeros([1, N-cur], dtype=x.dtype)
    return tf.concat([x, pad], axis=1)

def awgn_complex(x: tf.Tensor, snr_db: float, rng: np.random.Generator):
    x_np = x.numpy()
    p = np.mean(np.abs(x_np)**2).astype(np.float32)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = float(p / snr_lin)

    nI = rng.normal(0.0, np.sqrt(noise_var/2), size=x_np.shape).astype(np.float32)
    nQ = rng.normal(0.0, np.sqrt(noise_var/2), size=x_np.shape).astype(np.float32)
    n = nI + 1j*nQ
    y_np = x_np + n
    y = tf.convert_to_tensor(y_np.astype(np.complex64))
    return y, noise_var


# =========================
# Main
# =========================
if __name__ == "__main__":

    # --------- settings (edit here) ----------
    input_image_path = "cat (12).jpg"
    weight_path = "300T_0dB.pt"

    IMG_SIZE = 64
    jpeg_quality = 10

    # diffusion
    T = 1000
    t_start = 300

    # channel
    snr_db = 10.0
    seed = 0

    # superposition power split
    Pb = 0.7
    Pr = 0.3

    # digital
    mod_type = "qpsk"   # "bpsk" / "qpsk" / "16qam"
    ldpc_rate = "1/2"   # "1/2" / "3/4"

    # residual quantization (TX)
    rt_bits = 8
    rt_clip_k_sigma = 3.0

    OUT_DIR = os.path.join("outputs",
                           f"dig2_ms_{mod_type}_{ldpc_rate}_q{jpeg_quality}_t{t_start}_snr{int(snr_db)}_Pb{Pb}_Pr{Pr}")
    # ----------------------------------------

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if abs((Pb + Pr) - 1.0) > 1e-6:
        raise ValueError("Please keep Pb+Pr=1.0")

    # model
    model = UNet().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # schedules for multi-step reverse
    betas = make_beta_schedule(T=T, beta_start=1e-4, beta_end=0.0095)
    alphas, alpha_hat, beta_tilde = compute_schedules(betas)

    # ===== TX: source -> base bytes -> xb(tx local) -> residual -> diffuse -> quant -> bits =====
    x_pil = load_rgb_no_resize(input_image_path, IMG_SIZE)
    x = to_m11(pil_to_float01(x_pil)).astype(np.float32)

    # base "Q": JPEG bytes
    jpeg_bytes = jpeg_encode_bytes(x_pil, quality=jpeg_quality)

    # tx local xb for residual
    x_b_pil = jpeg_decode_bytes(jpeg_bytes)
    x_b = to_m11(pil_to_float01(x_b_pil)).astype(np.float32)

    r0 = (x - x_b).astype(np.float32)

    rt = forward_diffuse(r0, t=t_start, alpha_hat=alpha_hat, rng=rng)

    # residual quantize -> bits
    rt_q_u8, rt_q_meta = quantize_symmetric_uniform(rt, bits=rt_bits, clip_k_sigma=rt_clip_k_sigma)
    rt_bits_stream = u8_to_bits(rt_q_u8)

    # base bits stream
    base_bits_stream = bytes_to_bits_u8(jpeg_bytes)

    # ===== Digital modulate both branches =====
    modem = SionnaModem(mod_type=mod_type)

    sb, side_b = ldpc_modulate(base_bits_stream, rate=ldpc_rate, modem=modem)
    sr, side_r = ldpc_modulate(rt_bits_stream,   rate=ldpc_rate, modem=modem)

    Nb = int(sb.shape[1]); Nr = int(sr.shape[1]); N = max(Nb, Nr)
    sb = pad_to(sb, N)
    sr = pad_to(sr, N)

    # ===== Superposition + AWGN =====
    s = tf.cast(np.sqrt(Pb), tf.complex64) * sb + tf.cast(np.sqrt(Pr), tf.complex64) * sr
    y, noise_var = awgn_complex(s, snr_db=snr_db, rng=rng)

    # ===== RX-1: decode base from y =====
    base_bits_hat = demod_ldpc_decode(y, noise_var=noise_var, modem=modem, side=side_b)
    jpeg_bytes_hat = bits_to_bytes_u8(base_bits_hat)[:len(jpeg_bytes)]

    try:
        x_hat_pil = jpeg_decode_bytes(jpeg_bytes_hat)
        x_hat = to_m11(pil_to_float01(x_hat_pil)).astype(np.float32)
    except Exception:
        # if JPEG corrupted, fall back (still allow pipeline to run)
        print("[WARN] Decoded JPEG failed. Fallback to tx x_b as x_hat.")
        x_hat = x_b.copy()

    save_img_m11(x_hat, os.path.join(OUT_DIR, "RX_base_x_hat.png"))

    # ===== RX-2: SIC (remod base_hat then subtract) =====
    sb_hat, _ = ldpc_modulate(base_bits_hat, rate=ldpc_rate, modem=modem)
    sb_hat = pad_to(sb_hat, N)
    y_res = y - tf.cast(np.sqrt(Pb), tf.complex64) * sb_hat

    # ===== RX-3: decode residual from y_res =====
    rt_bits_hat = demod_ldpc_decode(y_res, noise_var=noise_var, modem=modem, side=side_r)
    num_q_u8 = rt_q_u8.size
    rt_q_hat_u8 = bits_to_u8(rt_bits_hat, num_u8=num_q_u8).reshape(rt_q_u8.shape)
    rt_hat = dequantize_symmetric_uniform(rt_q_hat_u8, rt_q_meta).astype(np.float32)

    # ===== RX-4: multi-step DDPM reverse to r0_hat (condition = RX x_hat) =====
    r0_hat_ms = reverse_residual_multistep_ddpm(
        model=model,
        r_t_rx=rt_hat,
        x_cond=x_hat,
        t_start=t_start,
        betas=betas,
        alphas=alphas,
        alpha_hat=alpha_hat,
        beta_tilde=beta_tilde,
        device=device,
        rng=rng,
    )

    save_img_m11(r0_hat_ms, os.path.join(OUT_DIR, "RX_residual_r0_hat.png"))

    # ===== Final combine =====
    x_final = np.clip(x_hat + r0_hat_ms, -1.0, 1.0).astype(np.float32)
    save_img_m11(x_final, os.path.join(OUT_DIR, "RX_final_x.png"))

    print("Done. Saved outputs to:")
    print("  ", OUT_DIR)
    print("Files:")
    print("  RX_base_x_hat.png")
    print("  RX_residual_r0_hat.png")
    print("  RX_final_x.png")
