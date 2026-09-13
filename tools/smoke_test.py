"""引擎冒烟测试：加载 IndexTTS2 并跑一次真实推理。

用法：
    .venv\\Scripts\\python.exe tools/smoke_test.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402

print("=" * 70)
print("IndexTTS-2.5 引擎冒烟测试")
print("=" * 70)

free, total = torch.cuda.mem_get_info(0)
print(f"GPU           : {torch.cuda.get_device_properties(0).name}")
print(f"显存          : {total/1024**3:.2f} GB 总量 / {free/1024**3:.2f} GB 空闲")
print(f"torch         : {torch.__version__}")
print(f"bf16 支持     : {torch.cuda.is_bf16_supported()}")
print()

from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

t0 = time.perf_counter()
print(">> 正在加载模型（use_bf16=True, use_qwen_emo=False）...")
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_bf16=torch.cuda.is_bf16_supported(),
    use_cuda_kernel=False,
    use_deepspeed=False,
    use_accel=False,
    use_torch_compile=False,
    use_qwen_emo=False,
)
load_sec = time.perf_counter() - t0
print(f"\n>> 模型加载完成，耗时 {load_sec:.1f}s")

alloc = torch.cuda.memory_allocated(0) / 1024**3
resv = torch.cuda.memory_reserved(0) / 1024**3
free, total = torch.cuda.mem_get_info(0)
print(f">> 显存占用: allocated={alloc:.2f}GB reserved={resv:.2f}GB "
      f"剩余空闲={free/1024**3:.2f}GB / {total/1024**3:.2f}GB")

# 统计参数量
n_gpt = sum(p.numel() for p in tts.gpt.parameters())
n_s2mel = sum(p.numel() for p in tts.s2mel.parameters())
n_codec = sum(p.numel() for p in tts.semantic_codec.parameters())
n_sem = sum(p.numel() for p in tts.semantic_model.parameters())
n_big = sum(p.numel() for p in tts.bigvgan.parameters())
print(f">> 参数量: GPT={n_gpt/1e6:.1f}M  w2vBERT={n_sem/1e6:.1f}M  "
      f"codec={n_codec/1e6:.1f}M  s2mel={n_s2mel/1e6:.1f}M  bigvgan={n_big/1e6:.1f}M")
print(f">> 合计 {(n_gpt+n_s2mel+n_codec+n_sem+n_big)/1e9:.3f} B")

# LoRA 可行性探测
print("\n>> GPT2 主干可注入 LoRA 的 Linear 层：")
targets = {}
for name, mod in tts.gpt.named_modules():
    if isinstance(mod, torch.nn.Linear):
        leaf = name.split(".")[-1]
        targets.setdefault(leaf, 0)
        targets[leaf] += 1
for k, v in sorted(targets.items(), key=lambda x: -x[1]):
    print(f"     {k:<24} x{v}")

print("\n>> CFM(DiT) 主干可注入 LoRA 的 Linear 层：")
cfm_targets = {}
for name, mod in tts.s2mel.models["cfm"].named_modules():
    if isinstance(mod, torch.nn.Linear):
        leaf = name.split(".")[-1]
        cfm_targets.setdefault(leaf, 0)
        cfm_targets[leaf] += 1
for k, v in sorted(cfm_targets.items(), key=lambda x: -x[1])[:14]:
    print(f"     {k:<28} x{v}")

print("\n>> 开始推理...")
prompt = "examples/voice_01.wav"
if not os.path.isfile(prompt):
    print(f"!! 参考音频不存在: {prompt}")
    sys.exit(1)

out = "outputs/_smoke_test.wav"
t0 = time.perf_counter()
res = tts.infer(
    spk_audio_prompt=prompt,
    text="大家好，欢迎使用 IndexTTS 二点五，这是一次引擎冒烟测试。",
    lang="ZH",
    output_path=out,
    verbose=False,
    max_text_tokens_per_segment=120,
    duration_factor=1.0,
    do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
    num_beams=3, repetition_penalty=10.0, length_penalty=0.0,
    max_mel_tokens=1500,
)
infer_sec = time.perf_counter() - t0

print(f"\n>> 推理返回: {res}")
if os.path.isfile(out):
    import soundfile as sf
    info = sf.info(out)
    dur = info.duration
    print(f">> 输出文件: {out}")
    print(f">> 时长 {dur:.2f}s  采样率 {info.samplerate}  声道 {info.channels}")
    print(f">> 推理耗时 {infer_sec:.2f}s   RTF = {infer_sec/dur:.3f}")
else:
    print("!! 未生成输出文件")
    sys.exit(1)

peak = torch.cuda.max_memory_allocated(0) / 1024**3
free, total = torch.cuda.mem_get_info(0)
print(f"\n>> 峰值显存 allocated = {peak:.2f} GB")
print(f">> 结束时空闲 {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")
print("\n" + "=" * 70)
print("冒烟测试通过 ✅")
print("=" * 70)
