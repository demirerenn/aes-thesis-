# HANDOFF — AES Tezi Projesi

**Hazırlanma tarihi:** 2026-04-15
**Hazırlayan:** Cowork (Claude Opus 4.6) → Claude Code (GX10) geçişi için
**Hedef:** Yeni Claude Code session'ı 30 saniyede tam bağlamı edinsin.

> **İlk komutun şu olsun:** `git log --oneline -10 && cat docs/HANDOFF.md && cat README.md`
> Sonra okuma sırası: `docs/HANDOFF.md` → `ADR_001_rev2_Addendum_v1.docx` → `Agent_Sistem_Tasarimi_v1.2.docx` → son olarak `Literatur_Raporu_v1.docx` (referans).

---

## 1. Proje Tek Cümlede

ASAP 1.0 + ASAP 2.0 (PERSUADE) veri setlerinde **QWK ≥ 0.80** hedefleyen, **transformer-tabanlı (DeBERTa-v3-large / Longformer) Automated Essay Scoring** sistemi. Eğitim ve karar süreci **LangGraph üzerine kurulmuş 16-agent'lı bir orkestra** ile yönetilir. Donanım: **ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell, ARM64, 128 GB unified memory)**. Çıktı: bir tez + reproducible artifact zinciri.

---

## 2. Mevcut Durum (Snapshot)

| Konu | Durum |
|------|-------|
| ADR-001-rev1 (orijinal mimari karar) | Onaylı (Agent_Sistem_Tasarimi_v1.2 ile uyumlu) |
| ADR-001-rev2 Addendum (Pilot Phase) | Onaylı, kod karşılığı tamamlandı (commit `5276ad8`) |
| Docker katmanı | NGC PyTorch 25.12-py3 base, build pass'lı (commit `9312baf` sonrası) |
| Trainer image | `aes-thesis-trainer:latest` (NGC 25.12 üzerine layer) |
| Sanity check (cuda avail, bf16, flash_attn) | **BEKLEMEDE** — handoff anında doğrulanacak |
| Pilot smoke test | Henüz çalışmadı |
| Pilot full run (3 seed, ~1.5 saat) | Henüz çalışmadı |
| Evaluator Agent (deterministik node) | Implementasyon tamam (commit `1f0274d`), synthetic data ile test edildi (QWK=0.9067 → decision=go) |
| Geriye kalan 14 agent node | Henüz implement edilmedi (Architect, Feedback Strategist, Thesis Writer, Fairness Auditor, vb.) |
| MLflow tracking | Compose'da hazır, henüz çalıştırılmadı |
| GitHub repo | `https://github.com/demirerenn/aes-thesis-` — tek branch (`main`), 8 commit |

---

## 3. ADR-001-rev2 Pilot Protokolü — Ne Yapacağız?

Orijinal plan **5-fold CV × 3 seed** idi (≈ 7 saat). Tez sahibi (Eren) iki kademeli protokole geçmek istedi:

**Pilot Phase (önce):**
- Sabit stratified split **70/15/15** (sklearn `train_test_split`, `essay_set × class_idx` üzerinde stratify)
- **3 training seed** (42, 123, 2024)
- ~1.5 saat (5-fold'a göre 5x hızlı)
- Çıktı: dev QWK + test QWK + per-sample `preds-*.csv`

**Gating (Evaluator Agent — `decide_gating()`):**

| QWK (dev mean) | Karar | Sonraki Agent |
|----------------|-------|---------------|
| ≥ 0.82 | **GO** — ileriye geç (sonraki bucket / tam CV'ye) | thesis_writer |
| 0.78–0.82 | **ITERATE-A/B** — hyperparameter veya loss ablasyon | feedback_strategy |
| 0.72–0.78 | **REVISE** — mimari revizyon (backbone/loss değişimi) | architect |
| < 0.72 | **ROLLBACK** — kök-neden analizi (data leak, tokenizer, loss bug) | architect |

Pilot başarılıysa (QWK ≥ 0.82) **tam 5-fold CV**'ye geçmeden tez yayımlanabilir; başarısızsa CV ile genişletip karşılaştırırız.

**Detaylar:** `ADR_001_rev2_Addendum_v1.docx` (10 bölüm, 97 paragraf).

---

## 4. Sıradaki Adımlar (Sırasıyla)

### 4.1 ŞU AN — Sanity Check
Build başarılıysa ilk komut:

```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -c "import torch, flash_attn, transformers; \
  print('GPU         :', torch.cuda.get_device_name(0)); \
  print('torch       :', torch.__version__); \
  print('cuda build  :', torch.version.cuda); \
  print('cuda avail  :', torch.cuda.is_available()); \
  print('bf16        :', torch.cuda.is_bf16_supported()); \
  print('flash_attn  :', flash_attn.__version__); \
  print('transformers:', transformers.__version__)"
```

**Yeşil kriter:** `cuda avail: True`, `bf16: True`, `GPU: NVIDIA GB10`. Bu üçü tutarsa pilot'a geç. Tutmazsa **driver/CUDA mismatch** araştır.

### 4.2 Pilot Smoke Test (3-5 dk)
1 seed × 1 epoch × tiny batch. Pipeline crash etmiyor mu kontrolü:

```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -m src.scripts.train_baseline \
    --config configs/baseline_asap1_p2.yaml \
    --mode pilot --smoke
```

QWK değeri anlamsız olacak. Sadece "no crash" + `summary_*.json` üretildi mi bakılır.

### 4.3 Pilot Full Run (~1.5 saat)
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -m src.scripts.train_baseline \
    --config configs/baseline_asap1_p2.yaml \
    --mode pilot
```

Çıktı: `runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0/` altında:
- `summary_*.json` — fold-level metrics + bootstrap CI
- `preds-pilot-s{42,123,2024}.csv` — per-sample predictions
- `ckpt-pilot-s*.pt` — checkpoint'ler

### 4.4 Evaluator Agent → Rapor
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -c "from pathlib import Path; \
  from src.agents.nodes.evaluator import EvaluatorInput, evaluate_run; \
  d = evaluate_run(EvaluatorInput( \
    run_dir=Path('runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0'), \
    config_path=Path('configs/baseline_asap1_p2.yaml'), \
    template_dir=Path('src/agents/templates'))); \
  print(d)"
```

Çıktı: `runs/.../report_*.md` (Jinja2 ile render'lanmış, 9 bölümlü tez-grade rapor).
Dönüş dict: `{decision, next_agent, qwk_mean, ...}`.

### 4.5 Karar Yoluna Göre Dallanma
- **GO** → ASAP 1.0 prompt 2 başarılı, sonraki bucket (prompt 1, 3, …) veya direkt tez yazımına geç
- **ITERATE-A** → LR/epoch/batch grid search (yeni Feedback Strategist agent — implement edilecek)
- **ITERATE-B** → Loss ablasyon (CORN ↔ CORAL ↔ MSE+rank ↔ label smoothing)
- **REVISE** → Architect agent: backbone değişimi (DeBERTa → Longformer veya RoBERTa-large)
- **ROLLBACK** → Data leak / tokenizer / loss bug avı

### 4.6 Tez Yazımı (Thesis Writer Agent)
Henüz implement edilmedi. ADR-001-rev2 §6: rev0 sonuçları kabul edilirse tez bölümlerini otomatik üretir.

---

## 5. Bilinen Gotcha'lar (Öğrenilmiş Acılar)

1. **`docker-compose.yml` ve `docker/Dockerfile` aynı `NGC_TAG` değerini içermeli.** Compose'un `args` bloğu Dockerfile ARG'ı **ezer**. Birini değiştirip ötekini unutursan eski sürümle build edersin. (Bugün 25.02 → 25.12 geçişinde tam 4 saat kaybettik.)

2. **PyPI `torch`, `torchvision`, `flash-attn` PAKETLERİNİ KURMA.** NGC container'ı bunları zaten Blackwell sm_100 için optimize derlemiş olarak getiriyor. PyPI wheel'leri override ederse `torch.cuda.is_available()` False döner ve `libc10_cuda.so` linkleme hatası alırsın. `requirements.txt`'nin başındaki yorum bloğunda yazıyor — okumadan paket ekleme.

3. **`weights-and-biases` PyPI'de yoktur.** Doğru paket adı `wandb`. (Bu da ilk Docker build'i yaktı.)

4. **`flash-attn` kaynaktan derleme ~30 dk sürer.** Eğer build'in 5/8 adımında `Building wheel for flash-attn (setup.py): still running...` görüyorsan **dur, requirements.txt'yi kontrol et** — flash-attn satırı kalmış olabilir, kurmuyor olmamız gerekiyor.

5. **GX10 driver: 580.x, NGC 25.02 ile uyumsuz.** 25.12 ile tutuyor. Eğer 25.02'ye geri dönmek zorunda kalırsan `cuda avail: False` problemi tekrarlar.

6. **`docker volume rm` agresif.** Şu volume'lar **korunmalı**:
   - `docker_hf-cache` (HuggingFace model cache, AES tezi için)
   - `open-webui` (kullanıcının Open-WebUI sohbet history'si)

7. **Bracketed paste mode terminal'de komutu bozabilir.** `^[[200~docker` gibi prefix görürsen `printf '\e[?2004l'` ile kapat.

8. **ASAP 2.0 CSV (198 MB) GitHub 100MB limitini aşıyor.** `.gitignore`'da `data/ASAP2_train_sourcetexts.csv` var. Veriyi flash disk ile manuel transfer.

9. **Author attribution diskuruna dikkat.** Tüm commit'ler `demirerenn` (Eren) tarafından imzalanmalı. Agent commit yapmamalı (audit trail temizliği).

---

## 6. Repository Layout

```
~/aes-thesis/
├── README.md                                # high-level proje overview
├── ADR_001_*.docx                           # mimari kararlar (ADR-001 v1, v1.1, rev1, rev2)
├── Agent_Sistem_Tasarimi_v1.2.docx          # 16-agent LangGraph mimarisi
├── Literatur_Raporu_v1.docx                 # AES literatür taraması (Taghipour 2016, R²BERT 2020, vb.)
├── EDA_Raporu_v1.docx                       # ASAP 1+2 exploratory data analysis
├── PeerReview_ADR001_v1.docx                # peer-review notları
├── Report_Template_v1.md                    # rapor şablonu (run_report.md.j2'nin kaynağı)
├── requirements.txt                         # PyPI paketleri (torch/flash-attn YOK!)
├── configs/
│   └── baseline_asap1_p2.yaml              # ASAP1 prompt-2 pilot config
├── docker/
│   ├── Dockerfile                          # NGC 25.12-py3 base
│   ├── docker-compose.yml                  # trainer + mlflow services
│   └── entrypoint.sh                       # banner + nvidia-smi
├── data/                                    # ASAP1 (committed) + ASAP2 (gitignored, flash transfer)
├── eda_reports/                             # EDA çıktıları (HTML, plots)
├── runs/                                    # training output (gitignored)
└── src/
    ├── aes/
    │   ├── data.py                         # ASAP1/2 loaders, fixed_split, stratified_folds
    │   ├── training.py                     # Trainer + TrainConfig + evaluate_df
    │   ├── metrics.py                      # qwk, bootstrap_ci, pearson, spearman, macro_f1
    │   └── utils.py                        # set_seed, capture_env, data_hashes
    ├── scripts/
    │   └── train_baseline.py               # CLI driver (--mode pilot|cv|auto)
    └── agents/
        ├── nodes/
        │   ├── __init__.py
        │   └── evaluator.py                # IMPLEMENTED — deterministik gating + report rendering
        ├── templates/
        │   └── run_report.md.j2            # Jinja2 thesis-grade report template
        └── (stubs to be added)             # architect.py, feedback_strategy.py, thesis_writer.py, ...
```

---

## 7. Commit History (commit'ler "ne yaptık" hikayesi)

```
9312baf  fix(docker): align compose NGC_TAG with Dockerfile (25.02 -> 25.12)
110c2dd  docs(requirements): make NGC base-tag note version-agnostic
1f0274d  feat(agents): Evaluator Agent with Jinja2 report template
5276ad8  feat(training): pilot phase support (ADR-001-rev2)
d28084d  chore(docker): bump NGC base image 25.02 -> 25.12-py3
0e57061  fix: drop torch/torchvision/flash-attn (NGC provides them)
f5969ae  fix: wandb package name (was invalid weights-and-biases)
9e04b6d  init: AES thesis scaffold (ADR-001-rev2 pilot protocol + Docker)
```

Yeni commit'ler için **conventional commits** kullan: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`. Tez peer-review'ı için audit trail değerli.

---

## 8. Tasarım İlkeleri (Tez Yönergesinden — değişmez)

1. **Sadece transformer-tabanlı modeller** (BERT, DeBERTa, RoBERTa, Longformer). Klasik ML veya >1B param dev LLM kullanma. Görev/veriye göre **gerekçeli seçim** yap (ihtiyaca göre regression vs classification, MSE vs CORN/CORAL).
2. **Literatür-bilgili mimari kararlar.** Her büyük seçim ADR ile gerekçelendirilmeli.
3. **Tek model her şeyi çözmek zorunda değil.** Essay tipine/uzunluğuna göre farklı backbone seçilebilir; bu literatür + sonuçlardan türetilmeli.
4. **Her training run benzersiz isimlendirilmeli.** `<dataset>-<prompt>-<backbone>-<loss>-<phase>-<rev>` şablonu (örn `asap1-p2-deberta_v3_lg-corn-pilot-rev0`).
5. **Maliyetten önce başarı.** Ücretli/açık kaynak farketmez, **performans** öncelik.
6. **Sistem dinamik.** Sonuçlara bakıp mimari/hyperparameter/kod güncellemesi yapabilmeli.
7. **GX10 donanım kısıtlarını ciddiye al.** 128 GB unified memory, sm_100, ARM64.
8. **Profesyonel kod.** Black + ruff disiplini, type hints, docstrings, dataclass'lar.
9. **Bu bir tez.** Her şey raporlanmalı, izlenebilir olmalı, peer-review'a dayanmalı.

---

## 9. Hızlı Komut Cheat-Sheet

```bash
# Sanity check
docker compose -f docker/docker-compose.yml run --rm trainer python -c "..."

# Smoke test
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -m src.scripts.train_baseline --config configs/baseline_asap1_p2.yaml --mode pilot --smoke

# Full pilot (~1.5 saat)
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -m src.scripts.train_baseline --config configs/baseline_asap1_p2.yaml --mode pilot

# Evaluator
docker compose -f docker/docker-compose.yml run --rm trainer \
  python -c "from pathlib import Path; from src.agents.nodes.evaluator import EvaluatorInput, evaluate_run; \
  print(evaluate_run(EvaluatorInput(run_dir=Path('runs/<NAME>'), config_path=Path('configs/baseline_asap1_p2.yaml'), template_dir=Path('src/agents/templates'))))"

# MLflow UI (background)
docker compose -f docker/docker-compose.yml up -d mlflow
# → http://localhost:5000

# Logs
docker compose -f docker/docker-compose.yml logs -f trainer

# GPU canlı
watch -n 2 nvidia-smi
```

---

## 10. Açık Sorular / İleri Çalışma

- **Ollama backend alternatifi:** GX10'da `ollama` + `open-webui` zaten kurulu. Tezin "Future Work" bölümünde, agent orchestration için Anthropic API yerine **local Llama 3.1 70B / Qwen 2.5 72B** alternatifi raporlanabilir. Maliyet sıfır, network bağımsız, reproducibility mükemmel.
- **Fairness Auditor agent:** ASAP 2.0 PERSUADE'de demografik etiketler var. Bu agent grup-arası QWK farkını 0.10 eşiğine göre denetleyecek (ADR §5). Henüz implement edilmedi.
- **Image digest pinning:** ADR-001-rev2 Annex B, NGC 25.12 image digest'ini istiyor. Sanity check yeşil dönünce `docker inspect aes-thesis-trainer:latest --format '{{.Id}}'` ile alıp ADR'a ekleyeceğiz.
- **`.gitattributes` ekle:** Windows ↔ Linux CRLF/LF uyarılarını susturmak için. Tek satır: `* text=auto eol=lf`. Şimdi yapmadık çünkü öncelik değildi.

---

*Hazırlayan: Cowork (Claude Opus 4.6) — son görev: smoke test öncesi handoff.*
*Devralan Claude Code instance'ına: hoş geldin. Eren'in tez zaman çizelgesi sıkı, fikirlerini açıkça paylaş, gereksiz yere uzatma, sandbox dance'ı yok artık — direkt git/docker/python komutlarını kendin çalıştır.*
