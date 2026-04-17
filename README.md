# Agent System — Otomatik Essay Puanlaması (Tez Projesi)

Transformer tabanlı AES için dinamik, çok-ajanlı sistem. Hedef metrik **QWK ≥ 0.80**, veri setleri ASAP 1.0 (Kaggle 2012) + ASAP 2.0 / PERSUADE (Kaggle 2024), donanım **ASUS Ascent GX10** (NVIDIA GB10, 128 GB unified).

## Projeyi İnceleme Rehberi

Bu projeyi ilk kez inceleyen biri aşağıdaki sırayı takip etmelidir:

### Aşama 1 — Bağlam: "Bu proje ne?" (~15 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 1 | `README.md` | Bu dosya — proje özeti, yapı, kurulum |
| 2 | `Literatur_Raporu_v1.docx` | AES alanı, SOTA modeller (R²BERT, PAES), QWK metriki |
| 3 | `EDA_Raporu_v1.docx` | ASAP veri seti analizi — prompt dağılımları, essay uzunlukları, sınıf dengesizliği |

### Aşama 2 — Kararlar: "Mimari neden böyle?" (~20 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 4 | `ADR_001_Ilk_Mimari_Karar_v1.1_rev1.docx` | İlk mimari karar: backbone (DeBERTa-v3-large), loss (CORN), eğitim stratejisi |
| 5 | `ADR_001_rev2_Addendum_v1.docx` | Pilot faz protokolü, gating eşikleri (GO/ITERATE/REVISE/ROLLBACK) |
| 6 | `docs/ADR_001_rev3_addendum.md` | İki-rejim sistemi (Rejim-A/B), literatür düzeltmesi, multi-prompt geçiş |

### Aşama 3 — Agent Sistemi: "Orkestrasyon nasıl çalışıyor?" (~20 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 7 | `Agent_Sistem_Tasarimi_v1.2.docx` | 16 agent mimarisi, görev tanımları, pipeline akışı |
| 8 | `src/agents/state.py` | Paylaşılan state şeması — AESState, Artifact, Decision TypedDict'leri |
| 9 | `src/agents/graph.py` | LangGraph DAG — 16 node, routing, LLM_ROUTING (8 Opus + 5 Sonnet + 2 GPT + 1 Gemini) |
| 10 | `src/agents/llm_factory.py` | Multi-provider LLM client factory — AGENT_CONFIG (system prompt'lar), `invoke_agent()`, `make_agent_node()` |
| 11 | `src/agents/nodes/evaluator.py` | Deterministik Evaluator — QWK gating, confusion matrix, Jinja2 rapor üretimi |

### Aşama 4 — Eğitim Kodu: "Model nasıl eğitiliyor?" (~30 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 12 | `src/aes/data.py` | ASAP veri yükleme, EssayDataset, stratified split, prompt-level ayrım |
| 13 | `src/aes/models.py` | DeBERTa-v3 / Longformer + regression head, sigmoid [0,1] çıktı |
| 14 | `src/aes/losses.py` | CORN, CORAL, MSE+rank kayıp fonksiyonları |
| 15 | `src/aes/metrics.py` | QWK hesaplama, bootstrap CI, per-prompt aggregation |
| 16 | `src/aes/training.py` | Trainer sınıfı — BF16, grad accumulation, per-prompt denormalize, multi-prompt eval |
| 17 | `src/scripts/train_baseline.py` | Ana çalıştırma scripti — pilot/CV modları, resume-on-crash, auto-Evaluator |

### Aşama 5 — Deneyler: "Hangi config'ler çalıştırıldı?" (~15 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 18 | `configs/baseline_asap1_p2.yaml` | İlk pilot — P2-only, DeBERTa-v3-large, CORN loss |
| 19 | `configs/asap1_allprompts_mse_rank_rejimB_confirm.yaml` | 8-prompt confirmatory — 3 seed × 8 epoch, MSE+rank |
| 20 | `configs/asap1_allprompts_mse_rank_iter_b_dropout02.yaml` | Dropout 0.2 ablasyon (negatif sonuç) |

### Aşama 6 — Altyapı: "GX10'da nasıl çalıştırılıyor?" (~10 dk)

| # | Dosya | Açıklama |
|---|-------|----------|
| 21 | `docker/Dockerfile` | NGC PyTorch base image, ARM64 (aarch64) + CUDA 12.6 |
| 22 | `docker/docker-compose.yml` | Servisler (trainer + mlflow), GPU, 16 agent env var pass-through |
| 23 | `.env.example` | API key yapılandırma şablonu (Anthropic + OpenAI + Google) |
| 24 | `docs/HANDOFF.md` | Geçiş rehberi, bilinen sorunlar, 9 kritik gotcha |

## Proje Yapısı

```
Agent System/
├── configs/                            # Eğitim config'leri (YAML)
│   ├── baseline_asap1_p2.yaml          # Pilot: P2-only, CORN loss
│   ├── asap1_allprompts_mse_rank_rejimB_confirm.yaml  # 8-prompt confirmatory
│   └── asap1_allprompts_mse_rank_iter_b_dropout02.yaml # Dropout ablasyon
├── data/                               # ASAP 1.0 TSV + ASAP 2.0 CSV
├── docker/
│   ├── Dockerfile                      # NGC PyTorch 25.12-py3 (ARM64)
│   └── docker-compose.yml              # trainer + mlflow servisleri
├── docs/
│   ├── ADR_001_rev3_addendum.md        # Güncel ADR (iki-rejim, multi-prompt)
│   └── HANDOFF.md                      # Geçiş rehberi
├── runs/                               # Checkpoints + summary JSON + rapor (git-ignored)
├── mlruns/                             # MLflow tracking (git-ignored)
├── src/
│   ├── aes/                            # Eğitim kütüphanesi
│   │   ├── data.py                     # ASAP loaders, stratified CV
│   │   ├── losses.py                   # CORN, CORAL, MSE+rank
│   │   ├── models.py                   # DeBERTa-v3 / Longformer + regression head
│   │   ├── metrics.py                  # QWK + bootstrap CI
│   │   ├── training.py                 # Trainer (BF16, grad-accum, multi-prompt eval)
│   │   └── utils.py                    # seed, hash, env capture
│   ├── agents/                         # LangGraph çok-ajanlı orkestrasyon
│   │   ├── graph.py                    # 16-node DAG, LLM_ROUTING, routing predicates
│   │   ├── llm_factory.py             # Multi-provider client (Anthropic+OpenAI+Google)
│   │   ├── state.py                    # AESState, Artifact, Decision şemaları
│   │   └── nodes/
│   │       └── evaluator.py            # Deterministik gating + Jinja2 rapor
│   └── scripts/
│       ├── train_baseline.py           # Ana eğitim scripti (resume + auto-Evaluator)
│       └── run_evaluator.py            # Standalone evaluator çağrısı
├── .env.example                        # API key şablonu
├── requirements.txt                    # Sabitlenmiş bağımlılıklar
├── Agent_Sistem_Tasarimi_v1.2.docx     # 16-agent mimari tasarım dokümanı
├── ADR_001_*.docx                      # Mimari karar kayıtları (v1, v1.1, rev1, rev2)
├── EDA_Raporu_v1.docx                  # Veri analizi raporu
├── Literatur_Raporu_v1.docx            # Literatür taraması
└── PeerReview_ADR001_v1.docx           # Peer review raporu
```

## Eğitim Sonuçları Özeti

| Deney | Config | QWK | Durum |
|-------|--------|-----|-------|
| Pilot P2-only (CORN) | `baseline_asap1_p2.yaml` | 0.697 | Baseline |
| 8-prompt smoke (MSE+rank) | `asap1_all_prompts_mse_rank_smoke.yaml` | 0.740 (2ep) | Pipeline doğrulama |
| 8-prompt confirmatory (3 seed) | `asap1_allprompts_mse_rank_rejimB_confirm.yaml` | 0.758 ± 0.002 | REVISE bandı |
| Dropout 0.2 ablasyon | `asap1_allprompts_mse_rank_iter_b_dropout02.yaml` | 0.750 | Negatif — capacity mismatch |
| R²BERT (literatür) | — | 0.794 | Hedef |

## Agent Sistemi — LLM Dağılımı

16 agent, 3 sağlayıcı, performans odaklı atama:

| Tier | Model | Agent Sayısı | Rol |
|------|-------|-------------|-----|
| Tier-1 | Claude Opus 4.7 | 8 | Stratejik muhakeme: Orchestrator, Research, Data Analyst, Architect, Training Engineer, Feedback Strategist, Thesis Writer, Review Reproducibility |
| Tier-2 | Claude Sonnet 4.6 | 5 | Yapılandırılmış analiz: Evaluator, Fairness Auditor, DevOps, Ops Monitor, Peer Coordinator |
| Tier-3 | GPT-5.3/5.4 + Gemini 3.1 | 3 | Kognitif çeşitlilik (peer review): Code Reviewer, Review ML Logic, Review Performance |

## Kurulum (GX10)

### Docker (önerilen)

```bash
# 1. Repo'yu GX10'a çek
git clone <your-repo-url> aes-thesis && cd aes-thesis

# 2. API key'leri konfigüre et
cp .env.example .env
# .env dosyasına ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY gir

# 3. Image'ı build et (~20 GB, ilk seferde NGC pull uzun sürer)
docker compose -f docker/docker-compose.yml build trainer

# 4. Ortam doğrulama
docker compose -f docker/docker-compose.yml run --rm trainer nvidia-smi
```

### Eğitim çalıştırma

```bash
# Pilot (1 seed × 8 epoch)
docker compose -f docker/docker-compose.yml run --rm trainer \
    python -m src.scripts.train_baseline \
        --config configs/asap1_allprompts_mse_rank_rejimB_confirm.yaml --mode pilot

# Agent DAG doğrulama (Mermaid diyagramı)
python -m src.agents.graph --dry-run
```

## Gating Eşikleri (Rejim-B — 8-prompt)

| Karar | QWK Aralığı | Aksiyon |
|-------|-------------|---------|
| GO | ≥ 0.82 | Sonraki faza geç (ASAP 2.0 veya 5-fold confirmatory) |
| ITERATE | 0.78–0.82 | Hiperparametre/loss ablasyon |
| REVISE | 0.72–0.78 | Mimari revizyon (backbone, head, strateji değişikliği) |
| ROLLBACK | < 0.72 | Kök-neden analizi (data leak, bug, tokenizer) |

## Yeniden Üretilebilirlik

- [x] ADR-001 rev1 peer review onaylı
- [x] ADR-001 rev3.2 — iki-rejim, multi-prompt protokolü
- [x] Veri SHA-256 hash'leri kayıt altında
- [x] 3-seed confirmatory tamamlandı (QWK 0.758 ± 0.002, CI [0.755, 0.760])
- [x] Resume-on-crash mekanizması aktif
- [x] Auto-Evaluator (run sonu otomatik gating + rapor)
- [ ] Agent node implementasyonu (Architect, Training Engineer — devam ediyor)
- [ ] DeBERTa-v3-base backbone scout
- [ ] ASAP 2.0 / PERSUADE protokolü (rev4)
