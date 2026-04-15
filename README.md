# Agent System — Otomatik Essay Puanlaması (Tez Projesi)

Transformer tabanlı AES için dinamik, çok-ajanlı sistem. Hedef metrik **QWK ≥ 0.80**, veri setleri ASAP 1.0 (Kaggle 2012) + ASAP 2.0 / PERSUADE (Kaggle 2024), donanım **ASUS Ascent GX10** (NVIDIA GB10, 128 GB unified).

## Proje Yapısı

```
Agent System/
├── configs/
│   └── baseline_asap1_p2.yaml          # ADR-001-rev1 §5 baseline config
├── data/                               # ASAP 1.0 TSV + ASAP 2.0 CSV
├── eda_reports/                        # Data Analyst visualizations
├── mlruns/                             # MLflow tracking (git-ignored)
├── runs/                               # Checkpoints + per-run summaries
├── src/
│   ├── aes/                            # Training library
│   │   ├── data.py                     # ASAP loaders, stratified CV
│   │   ├── losses.py                   # CORN, CORAL, MSE+rank
│   │   ├── models.py                   # DeBERTa-v3 / Longformer + ordinal heads
│   │   ├── metrics.py                  # QWK + fold-level bootstrap CI
│   │   ├── training.py                 # Single-run trainer (BF16, grad-accum)
│   │   └── utils.py                    # seed, hash, env capture
│   ├── agents/
│   │   ├── graph.py                    # LangGraph DAG (16 agents)
│   │   └── state.py                    # Shared state schema
│   └── scripts/
│       └── train_baseline.py           # CV driver + MLflow
├── Agent_Sistem_Tasarimi_v1.2.docx     # System architecture (thesis doc)
├── EDA_Raporu_v1.docx                  # Data Analyst output
├── Literatur_Raporu_v1.docx            # Research agent output
├── ADR_001_*_v1.1_rev1.docx            # Architect decision (approved)
├── PeerReview_ADR001_v1.docx           # Panel review
├── Report_Template_v1.md               # Evaluator report template
├── requirements.txt                    # Pinned dependencies
└── README.md                           # This file
```

## Kurulum (GX10 hedef)

### Önerilen yol — Docker (NVIDIA NGC PyTorch, ARM64-native)

ASUS Ascent GX10 (GB10 Grace Blackwell) için NGC PyTorch container'ı Flash-Attn-2, CUDA 12.6 ve BF16 kernel'leri önceden derlenmiş olarak getirir. Peer-review talep 13b-1 / 13c-1 bu katmanda çözülür.

```bash
# 1. Repo'yu GX10'a çek
git clone <your-repo-url> aes-thesis && cd aes-thesis

# 2. API key şablonunu kopyala (yalnızca agent wire-up sprint'i için zorunlu)
cp .env.example .env   # .env .gitignored

# 3. Image'ı build et (~20 GB, ilk seferde NGC pull uzun sürer)
docker compose -f docker/docker-compose.yml build trainer

# 4. Ortam bannerını gör (nvidia-smi + torch + flash_attn sürümleri)
docker compose -f docker/docker-compose.yml run --rm trainer nvidia-smi
```

### Alternatif — bare-metal venv (Docker istemezsen)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip pip-tools
pip install -r requirements.txt
pip-compile requirements.txt --resolver=backtracking -o requirements-lock.txt
```

## Baseline Eğitimini Çalıştırma

### Smoke test (dry-run, 1 seed × 1 epoch)

Docker:
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
    python -m src.scripts.train_baseline \
        --config configs/baseline_asap1_p2.yaml --mode pilot --smoke
```

Bare-metal:
```bash
python -m src.scripts.train_baseline \
    --config configs/baseline_asap1_p2.yaml --mode pilot --smoke
```

### Pilot Phase — fixed-split × 3 seed (ADR-001-rev2, ~1.5 saat GX10)

```bash
python -m src.scripts.train_baseline \
    --config configs/baseline_asap1_p2.yaml \
    --data-dir data \
    --mode pilot
```

Sabit stratified 70/15/15 bölünmesi; 3 training seed üzerinde ortalama QWK + std + test seti. **Gating eşikleri:**
- QWK ≥ 0.82 → en iyi config ile tam CV'ye geç
- 0.78–0.82 → 1–2 hiperparametre ablation + CV
- < 0.78 → mimari revizyon (CV'ye geçme)

### Phase 2 — Tam CV (pilot geçerse, 5-fold × 3 seed, ~6–8 saat GX10)

```bash
python -m src.scripts.train_baseline \
    --config configs/baseline_asap1_p2.yaml \
    --data-dir data \
    --mode cv
```

Çıktılar: `runs/<run_name>/ckpt-*.pt`, `runs/<run_name>/summary_<run_name>.json`, MLflow `mlruns/`.

## LangGraph Orkestrasyon İskelesi

Statik DAG doğrulaması (Mermaid diyagramı yazdırır):

```bash
python -m src.agents.graph --dry-run
```

LLM atamaları `LLM_ROUTING` sözlüğünde sabit; ortam değişkeni ile ezilebilir:

```bash
export AES_LLM_ARCHITECT=gpt-5.4-pro
python -m src.agents.graph --sprint 1
```

## Yeniden Üretilebilirlik Kontrol Listesi

- [x] ADR-001-rev1 onaylandı (ML Logic / Performance / Reproducibility)
- [x] Veri SHA-256 hash'leri kayıt altında (`ADR_001_*_v1.1_rev1.docx` Ek B)
- [x] Rapor şablonu tanımlı (`Report_Template_v1.md`)
- [ ] `requirements-lock.txt` GX10'da üretilecek (13c-1)
- [ ] 1-epoch dry-run torch.compile + Flash-Attn-2 doğrulaması (13b-1)
- [ ] MLflow experiment `aes-thesis` oluşturuldu

## İzlenecek Çıktı: İlk Baseline Run

**run_name:** `asap1-p2-deberta_v3_lg-corn-s42-rev0`

| Başarı Eşiği | QWK | Aksiyon |
|--------------|-----|---------|
| GO | ≥ 0.82 | SHORT kovasına genişlet |
| ITERATE | 0.78–0.82 | Hiperparametre arama + loss ablation |
| REVISE | 0.72–0.78 | Mimari revizyon (Feedback Strategist) |
| ROLLBACK | < 0.72 | Kök-neden analizi |

## Kaynaklar

Mimari ve kararlar için: `Agent_Sistem_Tasarimi_v1.2.docx`. Kayıp fonksiyonları ve backbone gerekçeleri: `Literatur_Raporu_v1.docx`. İlk mimari karar: `ADR_001_Ilk_Mimari_Karar_v1.1_rev1.docx`.
