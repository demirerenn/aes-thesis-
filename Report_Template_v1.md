# Deney Raporu Şablonu (v1)
*Evaluator Agent tarafından her baseline / ablation run'ı sonunda üretilir.*

> Bu şablon, peer-review talebi 13c-3 kapsamında ADR-001-rev1 ile onaylanmıştır.
> Her rapor MLflow run ID'si ile birlikte arşivlenir; adlandırma: `report_<run_name>.md`.

---

## 0. Meta-Bilgi

| Alan | Değer |
|------|-------|
| Run name | `<dataset>-<bucket>-<backbone>-<loss>-<seed>-<revN>` |
| MLflow Run ID | `<uuid>` |
| Tarih | `YYYY-MM-DD HH:MM TZ` |
| Donanım | ASUS Ascent GX10 / NVIDIA GB10 (128 GB unified) |
| Çalıştıran Agent | Training Engineer (#4) |
| Değerlendiren Agent | Evaluator (#8) |
| Rapor Sürümü | `rev<N>` |

## 1. Konfigürasyon Özeti

- **Dataset:** (ASAP 1.0 / ASAP 2.0 / combo) — prompt filtresi: `...`
- **N (train / val):** `<toplam>` / per-fold `<ortalama>`
- **Backbone:** `<HF model id>` — parametre sayısı: `<#M>`
- **Loss:** `<CORN / CORAL / MSE+rank / ...>`
- **max_seq_len:** `<int>` — batch: `<int>` (grad_accum `<int>`, efektif `<int>`)
- **LR / warmup / scheduler:** `<1e-5> / <0.1> / <linear>`
- **Epoch:** `<int>` — early-stop patience: `<int>`
- **Precision:** BF16 — torch.compile: `<on/off>` — Flash-Attn: `<on/off>`
- **Seed'ler:** `[s1, s2, s3]` — CV: stratified 5-fold
- **Veri hash'leri:** `asap1=<sha>`, `asap2=<sha>`
- **Env lock:** `requirements-lock.txt` hash: `<sha>`

## 2. Birincil Sonuçlar

### 2.1 QWK (birincil metrik)

| Fold | Seed s1 | Seed s2 | Seed s3 | Fold Ort. |
|------|---------|---------|---------|-----------|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| **Ortalama** | | | | **μ** |

**QWK = μ [α_lo, α_hi]** (%95 bootstrap CI, 1000 tekrar, fold-seviyesi resample)

### 2.2 İkincil Metrikler (tüm fold × seed ortalaması ± std)

| Metrik | Değer | Hedef | Durum |
|--------|-------|-------|-------|
| MAE | | — | |
| RMSE | | — | |
| Pearson r | | ≥ 0.85 | |
| Spearman ρ | | ≥ 0.85 | |
| Macro-F1 | | — | |

## 3. Confusion Matrix (Aggregate)

*5 fold × 3 seed = 15 değerlendirmenin birleşimi.*

```
           Tahmin
           1    2    3    4    5    6
Gerçek 1 [ .    .    .    .    .    . ]
       2 [ .    .    .    .    .    . ]
       3 [ .    .    .    .    .    . ]
       4 [ .    .    .    .    .    . ]
       5 [ .    .    .    .    .    . ]
       6 [ .    .    .    .    .    . ]
```

**Diagonal dominance:** `<%>` · **En sık hata çifti:** `(gerçek, tahmin) = (x, y)` — `<n>` adet

## 4. Hata Analizi

### 4.1 En Kötü 20 Tahmin (|tahmin − gerçek| sırasıyla)

| # | Gerçek | Tahmin | Δ | Kelime | Prompt | Essay önizleme (ilk 100 kar.) |
|---|--------|--------|---|--------|--------|-------------------------------|
| 1 | | | | | | |
| ... | | | | | | |

### 4.2 Hata Dağılımı

- Kısa (<200 kelime) essay'lerde MAE: `<x>`
- Uzun (>512 kelime) essay'lerde MAE: `<x>`
- Anonimize token yoğunluğu yüksek essay'lerde sapma: `<x>`

## 5. ASAP 2.0 — Fairness Alt-Raporu
*(Sadece ASAP 2.0 run'larında doldurulur — aksi halde "N/A".)*

| Demografik Grup | N | QWK | Δ (genel ort.) | Sinyal Bias (μ tahmin − μ gerçek) |
|-----------------|---|-----|----------------|-----------------------------------|
| race_ethnicity = White | | | | |
| race_ethnicity = Black/African American | | | | |
| race_ethnicity = Hispanic/Latino | | | | |
| ell_status = Yes | | | | |
| ell_status = No | | | | |
| gender = M | | | | |
| gender = F | | | | |
| economically_disadvantaged = Yes | | | | |
| economically_disadvantaged = No | | | | |

**Maksimum grup-arası QWK farkı:** `<x>` — Eşik `<0.10>`: `<geçti / geçmedi>`

## 6. Kaynak Tüketimi

- **Toplam wall-clock:** `<HH:MM:SS>`
- **Zirve GPU belleği:** `<GB>` / 128 GB
- **Toplam FLOPs (tahmini):** `<x>` PetaFLOP
- **Enerji (GX10 sensör):** `<kWh>` (opsiyonel)

## 7. Karşılaştırma

| Referans | QWK | Kaynak |
|----------|-----|--------|
| Taghipour & Ng 2016 (LSTM) | 0.761 | Literatür |
| R²BERT (Yang 2020) | 0.794 | Literatür |
| Bu run | `<μ>` | — |
| Önceki rev (`<run_name>`) | | MLflow |

## 8. Karar Önerisi (Stop / Go)

Aşağıdakilerden **bir** seçenek işaretlenir.

- [ ] **GO:** QWK ≥ 0.82 — ileriye genişlet (*sonraki kova / sonraki prompt*)
- [ ] **ITERATE-A:** 0.78 ≤ QWK < 0.82 — hiperparametre arama (LR, epoch, batch)
- [ ] **ITERATE-B:** 0.78 ≤ QWK < 0.82 — loss ablation (CORAL, MSE+rank, label smoothing)
- [ ] **REVISE:** 0.72 ≤ QWK < 0.78 — mimari revizyon (Feedback Strategist toplantısı)
- [ ] **ROLLBACK:** QWK < 0.72 — kök-neden analizi (data-leak, tokenizer uyumsuzluğu, loss bug)

**Gerekçe:** `<2–4 cümle, ölçülebilir gözlemlere dayalı>`

**Sorumlu bir sonraki Agent:** `<Training Engineer / Architect / Feedback Strategist>`

## 9. Ekler

- `confusion_matrix.png`
- `learning_curves.png` (train/val loss + QWK per epoch)
- `worst_predictions.csv` (En kötü 100 tahmin + meta)
- `fold_metrics.json` (makine-okunabilir metrik dökümü)
- `mlflow_run_url`: `<link>`

---

*Rapor onayı: Evaluator Agent + Feedback Strategist (Ajan #9) ikili onayı sonrası Tez Yazıcı Agent'a (Ajan #12) iletilir.*
