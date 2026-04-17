# Adım 1.2 — Smoke Training Pre-Flight Checklist

**Tarih:** 2026-04-17
**Ortam:** Sandbox pre-flight (x86_64, torch yok) → asıl koşu GX10'da
**Hedef:** `configs/baseline_asap1_p2.yaml` + `--smoke` ile 1 seed × 1 epoch × batch=4 çalıştırma

## Sandbox'ta doğrulanan (yeşil) ✓

| Kontrol | Sonuç |
|---|---|
| `yaml.safe_load(baseline_asap1_p2.yaml)` | OK — 8 top-level key |
| `load_asap1(data/training_set_rel3.tsv, prompts=[2])` | OK — 1800 satır |
| Skor aralığı (P2) | 1–6 (6 sınıf), class_idx 0..5, score_norm 0.0–1.0 |
| `fixed_split(df, 0.70/0.15/0.15, seed=42)` | OK — 1260 / 270 / 270 |
| Ortalama essay uzunluğu | ~2074 karakter / ~380 kelime (max_length=768 uygun) |
| QWK metriği (`src.aes.metrics.qwk`) | toy test: perfect=1.0, noisy=0.889 |
| `ast.parse()` — 7 training modülü | OK (syntax temiz) |
| `train_baseline.py` → `evaluate_run` otomatik tetikleniyor | OK (§6, satır 322-334) |

## Sandbox'ta doğrulanamayan (GX10'da test edilecek)

- PyTorch/CUDA stack — bu ortamda kurulamıyor
- `microsoft/deberta-v3-large` indirme (HuggingFace cache)
- MLflow `file:./mlruns` yazım izni
- bf16 destek (GB10 compute capability 10.0)

## GX10'da çalıştırılacak komut

```bash
cd ~/Agent\ System
# Ortam — NGC Docker tercih edilir (ARM64 uyumlu)
# docker compose up -d gx10-train  (projenin docker/ klasörüne göre)

python -m src.scripts.train_baseline \
  --config configs/baseline_asap1_p2.yaml \
  --smoke \
  --mode pilot 2>&1 | tee runs/_smoke_p2_$(date +%Y%m%d_%H%M).log
```

**Tahmini süre (GB10, bf16, batch=4, grad_accum=1):**
- 1260 train × 1 epoch ≈ 315 step
- DeBERTa-v3-large, max_length=768 → ~0.8 s/step on GB10 tahmini
- **Toplam: ~5–8 dk** (tokenization + warmup dahil)

## Yeşil ışık kriterleri (smoke)

1. Crash yok; tokenization + model load çalışıyor
2. `runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0/summary_*.json` oluşuyor
3. `fold_records[0]` içinde `qwk` NaN değil, [0.0, 1.0] aralığında
4. `runs/.../preds_dev.csv` (veya benzeri) raw integer tahminler içeriyor
5. `=== Evaluator ===` bölümü log'da görünüyor; `decision` ve `qwk_mean` basılıyor
6. `runs/.../evaluation_report.md` + `learning_curves.png` oluşuyor

## Smoke sonrası — gerçek pilot

Smoke yeşil olursa, `--smoke` bayrağını kaldırıp tam koşuya geç:

```bash
python -m src.scripts.train_baseline \
  --config configs/baseline_asap1_p2.yaml \
  --mode pilot 2>&1 | tee runs/_pilot_p2_$(date +%Y%m%d_%H%M).log
```

Bu 3 seed × 4 epoch × 1260 örnek → ~6–8 saat.

## Takip

Smoke sonucu paylaşıldığında:
- `fold_records[0].qwk` değerini kontrol edeceğiz (smoke'ta 0.4–0.7 arası beklenir)
- Evaluator'ın verdiği decision'ı doğrulayacağız (pre-condition fail mi, yoksa legit rollback mi)
- Eğer yeşilse → Adım 1.4 (Architect API testi) ile paralel olarak tam pilot başlar
