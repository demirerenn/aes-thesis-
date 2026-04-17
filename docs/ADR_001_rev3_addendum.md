# ADR-001 rev3 Addendum — Üç-Aşamalı Değerlendirme Protokolü ve Learning-Curve Temelli Gating Override

**Tarih:** 2026-04-16 (rev3.2 — literatür karşılaştırması düzeltildi, P2-spesifik eşikler, çok-prompt protokolü eklendi)
**Önceki revizyonlar:** rev3 ilk taslak → rev3.1 peer-review + inline patch → **rev3.2 literatür/eşik düzeltmesi**
**Statü:** Taslak — Eren onayı bekliyor
**Kapsam:** ADR-001-rev2 Addendum'unun gözden geçirilmiş hali.
**İlişkili artifact:** `runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0/` (pilot rev0 sonuçları)

---

## 1. Özet

ADR-001-rev2 iki aşamalı bir değerlendirme protokolü öngörüyordu:
1. **Pilot:** 3-seed sabit split, ~1.5 saat.
2. **Phase 2:** 5-fold CV × 3 seed, ~6-8 saat.

Rev3 bu protokolü **üç aşamaya** böler ve Evaluator Agent'ın mekanik gating kararının üzerine **learning-curve tabanlı insan yargısı katmanı** ekler. Motivasyon: rev2 ile yapılan pilot rev0, mekanik gating'in "rollback" verdiği ama eğrilerin "under-training" gösterdiği bir durumu ortaya çıkardı.

---

## 2. Motivasyon: Pilot Rev0 Gözlemleri

Pilot rev0 (`asap1-p2-deberta_v3_lg-corn-pilot-rev0`, 4 epoch × 3 seed):

| Seed | Best dev QWK | Best epoch | Son epoch QWK | Son epoch Δ (epoch 3→4) | Eğri tipi |
|------|-------------:|:----------:|--------------:|-------------------------:|:----------|
| 42   | 0.7237       | 4          | 0.7237        | **+0.0311**              | Monoton artış |
| 123  | 0.6804       | 2          | 0.6767        | +0.0061 (2→4 toplam: −0.0037) | Erken zirve, plato |
| 2024 | 0.6919       | 4          | 0.6919        | **+0.0189**              | Monoton artış |

**Özet metriği:** QWK mean = 0.6987, std (sample, ddof=1) = 0.0224.[^1]

[^1]: `summary_*.json`'daki `qwk_std` alanı `numpy.std(..., ddof=1)` ile hesaplanır (kod: `src/scripts/train_baseline.py` `obs.std(ddof=1)`). n=3 için sample std uygun; population std olsaydı ≈ 0.0183.

**Gözlemler:**

- **İki seed'de monoton yükseliş (s42, s2024):** Son epoch'ta anlamlı iyileşme — klasik under-training sinyali. Plato yok.
- **s123'te erken plato:** Epoch 2'de zirve (0.6804), 3'te 0.6706, 4'te 0.6767. Plato + noise. Loss monoton azalıyor (0.4029 → 0.2534 → 0.2037 → 0.1840) ama val_qwk 0.68 çevresinde titriyor. Klasik overfitting trajektorisi (val ↓ iken train ↑) değil, ama kapasite doygunluğu veya farklı optimizasyon dinamiği olarak okunabilir. Diğer iki seed ile aynı deseni paylaşmadığı için agregat sinyal "ağırlık under-training'de" kalır — ama s123'ün nüansı burada kayıt altına alınır, yumuşatılmaz.
- **Düşük seed-varyansı (std = 0.022):** Optimizasyon kararlı. Loss bug, tokenizer problemi veya data-leak sinyali yok.
- **Sistematik hata paterni (gerçek=3 → tahmin=4, 102 adet):** Hafif yukarı-bias, tutarlı.

**Mekanik gating karar:** `ROLLBACK` (QWK < 0.72).
**Kanıt-temelli gerçek karar:** `ITERATE-A` (epoch budget artırımı; ağırlık kanıtı under-training'de).

Rev2'nin gating'i bu ayrımı yapmıyordu — eşikler her durumu aynı şekilde "arıza" olarak görüyordu. Rev3 bu ayrımı override kuralları ile kodlaştırır (§4).

---

## 3. Değişen Karar: Üç-Aşamalı Protokol

Rev2'nin iki aşamalı protokolü üçe bölünür:

**Gerekçeler:**
1. Model-seçimi ve final-değerlendirme karıştırılmamalı. Rev2 tuning ile raporlamayı aynı protokole yıkıyordu.
2. Ablation'lar için fixed-split şart — data-split değişkenliği config sinyalini bulandırır.
3. Zaman ekonomisi: 5-fold × 3 seed = 15 run/ablation, 3-seed fixed = 3 run/ablation. ITERATE turları mümkün kılınıyor.

**Rev3 protokolü:**

| Faz | Protokol | Süre formülü[^2] | Amaç |
|-----|----------|:-----------------|------|
| **1. Pilot** | 3-seed fixed 70/15/15, epoch 4 | 3 × 4 × ~312 sn ≈ 3744 sn ≈ **1.0 sa net**; + ~25 dk ilk-run setup → ~1.5 sa wall-clock | İlk go/no-go + öğrenme eğrisi taraması |
| **2. ITERATE / REVISE** | 3-seed fixed (aynı split_seed) | 3 × e_cfg × ~345 sn (+ setup amortize) | Config tuning (epoch, LR, reg, loss, backbone) |
| **3. Confirmatory** | **5-fold × 3 seed** (sadece final config) | 15 × e_final × ~345 sn | Peer-review grade final raporlama |

[^2]: **Süre tahmininin tabanı:** ~312 sn = pilot rev0 **steady-state** per-epoch training wall-clock (ilk-run warmup, model download, tokenizer cache, CUDA context init hariç). İlk-run overhead ≈ 25 dk; bu nedenle pilot rev0'ın toplam wall-clock'u 3 × 4 × 312 ≈ 62 dk net + ~25 dk setup ≈ **~1.5 sa** olarak gözlemlendi. Aynı container'da yeniden çalıştırmada setup kısmen cache'li kalır (~5-10 dk'ya düşer). Faz 2/3 formülünde ~345 sn = ~312 sn + train-QWK inference eklentisi (§5, ~30 sn) + MLflow/JSON serialize (~3 sn). Faz 2 ITERATE-A (epochs=12) net: **3 × 12 × 345 ≈ 3.5 sa** + setup amortize ~15 dk → **~3.75 sa wall-clock**. Faz 3 aynı 12-epoch config: **15 × 12 × 345 ≈ 17-18 sa** net (setup 15×'de ihmal edilebilir oran). Rev3 ilk versiyonundaki "6-8 sa" tahmini 4-epoch varsayımına dayanıyordu ve hatalıydı — rev3.1'de düzeltildi.

**Faz 2 iç döngüsü:**
- Her ablation bağımsız `run_name` ve `configs/*_iter_<X>.yaml`.
- Split_seed **sabit** (pilot rev0 ile apples-to-apples karşılaştırılabilir).
- QWK ≥ 0.82 kriteri Faz 2'de de geçerli — bir ablation eşiği geçerse doğrudan Faz 3'e geçilir.

**Faz 3 çıktı zorunlulukları (peer-review grade):**
- Fold-level **bootstrap 95% CI**, 1000 resample (`bootstrap_ci_fold_level()` implement edildi, kullanılacak).
- **Per-seed ve per-fold QWK breakdown tablosu** (tez ekine direkt taşınır format).
- **Confusion matrix** + diagonal dominance + en sık hata çiftleri.
- **Length-bucket MAE** (very_short / short / medium / long / very_long).
- **Tüm run_name'ler, data SHA-256 hash'leri, env snapshot'ı** summary JSON'da.
- **Karşılaştırma tablosu (literatür-uyumlu protokol):** Aşağıdaki referans değerler **8-prompt ortalaması** olarak raporlanmıştır (her prompt için ayrı QWK hesaplanır ve ortalama alınır). Rev3.2 itibarıyla bizim karşılaştırmamız da aynı protokole çekilecek (§9).

  | Model | Protokol | P2 QWK | 8-prompt avg QWK |
  |-------|----------|:------:|:----------------:|
  | Taghipour & Ng 2016 (LSTM+CNN) | Per-prompt train/test, avg | ~0.68 | 0.761 |
  | R²BERT 2020 (BERT regression) | Per-prompt train/test, avg | **0.719** | 0.794 |
  | Pilot rev0 (bu run, DeBERTa-v3-large + CORN) | **P2-only** 70/15/15 | 0.6987 | — |

  **Kritik gözlem:** Pilot rev0'ın P2 QWK'si (0.697) R²BERT'in P2-spesifik performansına (0.719) yalnızca **0.022 uzakta** — "literatürün 0.07 altında" okuması rev3.1 ve öncesinde hatalıydı. "0.794 literatür tavanı" 8-prompt ortalaması; P2 ASAP1'deki zor prompt'lardan biri (en düşük P3:0.698 ile yakın). R²BERT per-prompt breakdown: P1:0.817, P2:0.719, P3:0.698, P4:0.845, P5:0.841, P6:0.847, P7:0.839, P8:0.744 (avg 0.794).

ASAP 2.0 / PERSUADE için ayrı bir confirmatory protokolü §9'da **rev4'e ertelendi** — ASAP1 final config belirlenmeden PERSUADE'nin protokolünü tasarlamak premature optimization.

---

## 4. Değişen Karar: Gating Override (Öğrenme-Eğrisi Yargısı)

Rev2'nin gating eşikleri (QWK 0.82 / 0.78 / 0.72) **8-prompt ortalaması** protokolüne bağlıdır — rev3.2'de P2-spesifik eşikler kalibre edildi.

### 4.0 Eşik Rejimleri (rev3.2'de eklendi)

Rev3.1'e kadar gating eşikleri (0.82 / 0.78 / 0.72) ASAP1 standart protokolünün **8-prompt ortalaması** kabul edilmişti. Ancak pilot rev0 yalnızca **P2** üzerinde değerlendirildi; R²BERT'in P2-spesifik performansı 0.719, yani 0.72 GO eşiği P2 için R²BERT'i geçmek anlamına geliyor — gerçekçi değil.

**Rev3.2 iki-rejim eşiği:**

| Rejim | Protokol | GO | ITERATE | REVISE | ROLLBACK |
|-------|----------|:--:|:-------:|:------:|:--------:|
| **Rejim-A** (P2-only, pilot/iter için) | Tek prompt, 3-seed fixed split | ≥ 0.72 | 0.66-0.72 | 0.60-0.66 | < 0.60 |
| **Rejim-B** (8-prompt, literatür-uyumlu) | Per-prompt QWK, sonra ortalanmış | ≥ 0.82 | 0.78-0.82 | 0.72-0.78 | < 0.72 |

**Rejim-A kalibrasyonu:** R²BERT P2=0.719 baz alınarak ±%1 tampon. GO "R²BERT-P2-civarı veya üstü" anlamına gelir. ITERATE "R²BERT'in P2'den %3-6 altı, kapanabilir fark". ROLLBACK "tanı gerekiyor" sinyali.
**Rejim-B kalibrasyonu:** Rev3.1 ile aynı — R²BERT 8-prompt avg 0.794 tavanından türetildi.

**Kullanım:** Faz 1-2'de Rejim-A geçerli (tek prompt'la hızlı iterasyon). Faz 3 confirmatory 8-prompt ortalamaya geçer → Rejim-B bağlayıcı.


### 4.1 Override Kuralları

**R1 — Under-training → ITERATE-A:**
Aşağıdaki üç koşul birlikte sağlanırsa mekanik karar "iterate-a"ya override edilir:

- (a) Seed'ler arası std(QWK) < 0.05.
- (b) En az bir seed'in son epoch'ta dev QWK artışı > 0.015.
- (c) Hiçbir seed'de klasik overfitting trajektorisi ("val QWK düşüyor **ve aynı anda** train QWK yükseliyor") yok.[^3]

[^3]: R1(c)'nin **formal doğrulaması** yalnızca `history.train_qwk` mevcut olduğunda yapılabilir. History yoksa (pilot rev0), (c) **loss trendi + val_qwk trajektorisi üzerinden dolaylı çıkarılır** ve **⚠ zayıf teyit** olarak işaretlenir. Rev3 §5 ile train_qwk sonraki tüm run'larda loglu; bu istisna sadece pilot rev0'a uygulanır.

**R2 — Overfitting → ITERATE-B:**
Aşağıdaki koşullar birlikte sağlanırsa:
- Ortalama train-val gap (`train_qwk − val_qwk`) ≥ 0.15.
- Val QWK plato'da: son 2 epoch |Δ| < 0.01.

Ek epoch değil **regularization / loss ablation** gerekir (dropout artırımı, label smoothing, CORN → CORAL, weight_decay artırımı).

**R3 — Yüksek varyans → ROLLBACK:**
std(QWK) ≥ 0.08 ise mean yüksek olsa bile şüphe edilir; data-leak / seeding / tokenizer tanıları çalıştırılır. Mekanik "GO" bile olsa R3 bağlayıcıdır.

### 4.2 ITERATE Taksonomisi (rev3.1'de netleştirildi)

Rev2'de `ITERATE-A` ve `ITERATE-B` yalnızca "aynı band'da iki varyant" anlamındaydı. Rev3 bunları **sebep-temelli** ayırır:

| Etiket | Neyi değiştirir? | Tetikleyici |
|--------|-------------------|-------------|
| **ITERATE-A** | Epoch budget, early_stop_patience | R1 (under-training) |
| **ITERATE-B** | Regularization (dropout, label smoothing, weight_decay, loss family) | R2 (overfitting) |
| **ITERATE-C** | LR grid, warmup ratio, optimizer (AdamW → Adafactor vb.) | Plato (R1 de R2 de yok; mekanik karar 0.78-0.82 aralığında) |
| **ITERATE-D** | Loss function ablation (CORN ↔ CORAL ↔ MSE+rank) | Çapraz-eksen; herhangi bir faz'da standalone veya diğerleriyle birlikte |

**0.78-0.82 band'ında öncelik sırası:** (1) R1 kontrolü → varsa A, (2) R2 kontrolü → varsa B, (3) hiçbiri yoksa → C, (4) ek exploration gerekirse → D.

### 4.3 Override Kullanım Kuralları

- **Override yalnızca mekanik karardan sapılırsa yazılı kayıt gerektirir.** Mekanik GO (QWK ≥ 0.82) doğrudan Faz 3'e geçer — override gereksiz. **Not:** Rev2 mekanik gating yalnızca GO / ITERATE / REVISE / ROLLBACK üretir; A/B/C/D alt-dalları §4.2'ye özgü sebep-temelli taksonomi olup yalnızca R1/R2 tetikleyicisiyle belirlenir (mekanik gating tanı yapmaz). Dolayısıyla mekanik "ITERATE" kararı (0.78-0.82 bandı) R1 tetikleyip ITERATE-A'ya gidilmesiyle kesişiyorsa **ayrı override kaydı gereksiz** — band + R1 eşleşmesi zaten aynı sonucu veriyor demektir.
- **Override formatı:** Bu addendum benzeri, her override için §X notu (hangi kural, hangi kanıt, hangi karar).
- **Override geri-alma kuralı:** Aynı override kuralı **ardışık iki Faz 2 turunda en fazla bir kez** uygulanır. İkinci turda aynı sinyal devam ediyorsa mekanik karar **bağlayıcıdır** ("R1 iki kere üst üste tetiklendiyse epoch budget yeterli değil, mimari/veri problemi olasılığı devreye girmiş" disiplini).

### 4.4 Pilot Rev0'a Uygulama

- **Rejim:** A (P2-only). Pilot rev0 prompts=[2] ile çalıştırıldı.
- **Mekanik karar (Rejim-A):** ITERATE — 0.66 ≤ 0.6987 < 0.72 (rev3.1'de yanlışlıkla Rejim-B uygulanmıştı; rev3.2 bu sınıflandırmayı geri almaktadır).
- **R1 koşulları:**
  - (a) std = 0.022 < 0.05 — **✓**
  - (b) s42 Δ = +0.031 > 0.015 — **✓**
  - (c) Train_qwk rev0'da logged değildi; dolaylı çıkarım: üç seed'de de loss monoton azalıyor, s42/s2024'te val_qwk yükseliyor, s123'te plato + noise (2→4 toplam Δ=−0.004, |Δ|<0.01 gürültü bandında). Klasik overfitting imzası (val ↓ + train ↑) gözlemlenmedi ama **(c) formal doğrulanamadı**. Footnote [^3] uyarınca **⚠ zayıf teyit**.
- **Override karar:** ITERATE-A — R1 tetiklendi, (c) zayıf teyitle.
- **Override sayacı (§4.3 kuralı için):** Pilot → iter_a geçişi, **1. kullanım**. Iter_a da başarısızsa ve R1 tekrar tetiklenirse, kural gereği mekanik karar (muhtemelen ROLLBACK/REVISE) bağlayıcı olur.

---

## 5. Değişen Karar: Per-Epoch Train-QWK Logging

Rev2 yalnızca `train_loss` ve `val_qwk` logluyordu. Rev3'te train QWK de her epoch ölçülür ve `summary_*.json → fold_records[i].history` altına yazılır.

**Gerekçe:**
- R2 override kuralı train-val gap'e doğrudan dayanır.
- Learning curve post-hoc analizi (Evaluator PNG render eder, §6).
- Literatür konvansiyonu — R²BERT, Taghipour & Ng ablation tabloları train/val birlikte raporluyor.

**Maliyet:** Epoch sonunda train seti üzerinde 1 inference pass (~30 sn/epoch) + MLflow log + JSON serialize overhead (~2-3 sn/epoch) = ~32-33 sn/epoch. 12 epoch × 3 seed = **~20-25 dk ek / run**. Kabul edilebilir.

**Implementasyon:** `src/aes/training.py` `Trainer.fit()`; `Trainer.history` attribute. `src/scripts/train_baseline.py` history'yi `fold_record`'a kopyalar, summary JSON'a otomatik yazılır.

---

## 6. Değişen Karar: Evaluator Agent'a Learning Curve Plot

Evaluator Agent `summary.fold_records[i].history` verisinden `runs/<run_name>/learning_curves_<run_name>.png` üretir:

- **Üst panel:** Seed başına train (kesikli) + val (düz) QWK eğrisi; gating eşikleri (0.72 / 0.78 / 0.82) yatay referans çizgileri.
- **Alt panel:** Seed başına `train_qwk − val_qwk` trajektorisi (R2 sinyali için doğrudan görsel).

**Implementasyon:** `src/agents/nodes/evaluator.py` `render_learning_curves()`, matplotlib Agg backend (headless container).

**Font/encoding notu:** NGC container'da sistem fontları sınırlı. Matplotlib default'u **DejaVu Sans** Latin genişletilmiş karakterleri (Türkçe dahil) kapsar — ek yapılandırma gerekmez. Rev4'te akademik özel font (Times New Roman vb.) istenirse Dockerfile'a `.ttf` kopyalama gerekir.

---

## 7. Bu Addendum'un Geriye Dönük Etkisi

- **Pilot rev0 çıktısı değişmez**, etiketi "pilot" kalır. Override retrospektif olarak §4.4'te kayıt altına alındı.
- **Pilot rev0 summary.json'da `history` alanı yoktur** (trainer o sırada loglamıyordu); R1(c)'nin pilot rev0 için ⚠ işaretlenmesinin teknik sebebi budur (§4.4, [^3]). Iter_a rev0 ve sonraki run'lar history ile zenginleşir.
- **`configs/baseline_asap1_p2.yaml`** pilot rev0 config'i olarak donduruldu. Yeni run'lar kendi config dosyasıyla çalışır (`configs/asap1_p2_iter_a.yaml` örneğin).

---

## 8. Sonraki Adımlar

1. `configs/asap1_p2_iter_a.yaml` ile ITERATE-A run'ını başlat (R1 override sonucu).
2. Tamamlanınca Evaluator çalıştırılır — ilk kez `learning_curves_*.png` üretilecek.
3. Sonuca göre karar matrisi:

**Rejim-A (P2-only, Faz 1-2):**

| İter_a sonucu | R1/R2 durumu | Sonraki adım |
|---------------|--------------|--------------|
| QWK ≥ 0.72 | — | Rejim-B'ye geçiş: full 8-prompt confirmatory (aynı config) |
| 0.66-0.72 | R2 tetiklendi | ITERATE-B (regularization) |
| 0.66-0.72 | R1/R2 yok, plato | ITERATE-C (LR grid) veya D (loss) |
| 0.60-0.66 | — | REVISE (backbone / sequence length / loss family) |
| < 0.60 | R1 aynen devam | §4.3 gereği mekanik bağlayıcı → **ROLLBACK** |
| < 0.60 | R3 (std yüksek) | **ROLLBACK** — tanı: data-leak / tokenizer / seeding |

**Rejim-B (8-prompt avg, Faz 3 confirmatory):**

| Sonuç | Aksiyon |
|-------|---------|
| QWK ≥ 0.82 | **Thesis-grade raporlama**, final confirmatory başarılı |
| 0.78-0.82 | ITERATE (ek config turu 8-prompt protokolünde) |
| 0.72-0.78 | REVISE |
| < 0.72 | ROLLBACK |

4. Phase 3 sonunda tez bölümü yazımı (Thesis Writer Agent — henüz implement edilmedi).

---

## 9. Çok-Prompt Protokolü (rev3.2'de eklendi)

**Rev3.1'e kadar:** Tüm gating kararları tek prompt (P2) üzerinden veriliyordu. Literatür karşılaştırması yanlış bir temelde (P2-only vs 8-prompt avg) yapıldı.

**Rev3.2 düzeltmesi:** ASAP1 tam 8-prompt protokolü confirmatory için bağlayıcı hale getirildi:

- **Veri yükleme:** `src/aes/data.py::load_asap1(prompts=None)` (mevcut, 8 prompt desteği var).
- **Skor normalizasyonu:** Her prompt farklı score-range'e sahip. `ASAP1_PROMPTS[pid].normalize(score)` → [0,1]. Eğitim [0,1] üzerinden, değerlendirme per-prompt denormalize edilip orijinal integer skor uzayında QWK hesaplanır.
- **Loss seçimi:** CORN 6-sınıf P2'ye özel. Çok-prompt için **MSE+rank** (regresyon head, mevcut `src/aes/losses.py::MSEPlusRankLoss`) default; rev4'te alternatif olarak per-prompt head denenebilir.
- **Metrik protokolü:**
  1. Prediction: model → sigmoid → [0,1] per-prompt range'e mapla → round → integer skor.
  2. Per-prompt QWK: her prompt için `cohen_kappa_score(weights='quadratic')`.
  3. Ortalama: 8 prompt'un QWK ortalaması (literatür konvansiyonu; pooled-QWK değil).
- **Eğitim/test bölme:** Literatürle uyum için prompt stratified 70/15/15 (her prompt'un dağılımı korunur).

**Faz 3 confirmatory bu protokole göre çalıştırılır.** Faz 2 ITERATE turları tek prompt'ta (P2) kalabilir — hızlı iterasyon için; final 8-prompt'a generalizasyon Faz 3'te teyit edilir.

---

## 9b. Kapsam Sınırı — ASAP 2.0 / PERSUADE Protokolü

Rev3 yalnızca **ASAP1** için protokolü tanımlar. ASAP 2.0 (PERSUADE) için protokol **rev4'e ertelenmiştir.** Yol haritası:

- **Protokol türü:** Cross-prompt hold-out CV (her run'da N−1 prompt train/dev, 1 prompt test). Rev2'deki stratified fold bu görev için uygun değil — PERSUADE'nin tez değeri cross-prompt generalization'da.
- **Ek metrik:** Demografik grup-bazlı QWK (cinsiyet, ELL statüsü, IEP statüsü, race/ethnicity). `src/aes/metrics.py::per_group_qwk` implement edildi; Fairness Auditor agent bu fonksiyonu kullanacak (henüz implement edilmedi).
- **Gate eşikleri:** ASAP1 final sonuçları kalibrasyon noktası — rev4 bunlara göre yazılacak.
- **Bağımlılıklar:** Fairness Auditor agent (rev4), ağırlıklı QWK alternatifleri (Taghipour convention).

**Gerekçe:** ASAP1 final config belirlenmeden PERSUADE protokolünü tasarlamak erken. ASAP1'in loss/backbone/seq_length seçimleri PERSUADE için başlangıç noktası olacak; rev4 bu başlangıçtan türetilecek.

---

## 10. İlgili Referanslar

- ADR-001 rev1 (orijinal): `ADR_001_Ilk_Mimari_Karar_v1.1_rev1.docx`
- ADR-001 rev2 Addendum (pilot protokol): `ADR_001_rev2_Addendum_v1.docx`
- Agent sistem tasarımı: `Agent_Sistem_Tasarimi_v1.2.docx`
- Pilot rev0 rapor: `runs/asap1-p2-deberta_v3_lg-corn-pilot-rev0/report_asap1-p2-deberta_v3_lg-corn-pilot-rev0.md`
- Peer-review geri bildirimi (rev3 → rev3.1 tetikleyicisi): 2026-04-16 tarihli Eren incelemesi (12 madde), conversation log.

---

## 11. Revizyon Notları

- **rev3 (2026-04-16 ilk taslak):** 3-aşamalı protokol, R1/R2/R3 override kuralları, train_qwk logging, learning curve plot.
- **rev3.1 (2026-04-16 peer-review işlendi):**
  - §2 s123 nüansı netleştirildi (plato + noise olarak ayrı kategorize); tablo Δ kolonu "epoch 3→4" net, s123 için 2→4 toplam eklendi.
  - §3 Phase 3 süre tahmini düzeltildi (6-8 sa → ~17-18 sa, 12-epoch final varsayımıyla) + süre formülü açık yazıldı.
  - §3 Faz 3 çıktı zorunlulukları listesi eklendi (bootstrap CI, per-seed/fold tablo, confusion, length bucket).
  - §4.1 R1(c) formülasyonu güçlendirildi ("ve aynı anda" eklendi); ⚠ işareti + footnote [^3] dolaylı çıkarım için.
  - §4.2 ITERATE taksonomisi A/B/C/D olarak sebep-temelli ayrıldı (eski ITERATE-B semantik çakışması çözüldü).
  - §4.3 override kapsam kuralı netleşti (sapma varsa kayıt; GO için gerekmez) + override geri-alma kuralı (ardışık iki tur limiti).
  - §4.4 pilot rev0 uygulaması ⚠ ile işaretlendi (c) için; override sayacı eklendi.
  - §5 maliyet dürüstçe 20-25 dk (MLflow + JSON overhead dahil).
  - §6 DejaVu Sans fallback notu.
  - §9 ASAP 2.0 kapsam-sınırı açık belgelendi (rev4'e ertelendi).
  - Footnote [^1] sample std (ddof=1) notu.
- **rev3.2 (2026-04-16 literatür/eşik düzeltmesi):**
  - §3 literatür karşılaştırma tablosu düzeltildi: R²BERT P2=0.719 (per-prompt) vs 0.794 (8-prompt avg) ayrımı eklendi. Pilot rev0 0.697 "literatürün 0.07 altında" değil, P2 için **0.022 uzakta**.
  - §4.0 eklendi: **Rejim-A (P2-only)** ve **Rejim-B (8-prompt avg)** iki-rejim eşiği. Rejim-A GO ≥ 0.72, ITERATE 0.66-0.72, REVISE 0.60-0.66, ROLLBACK < 0.60.
  - §4.4 pilot rev0 yeniden sınıflandırıldı: Rejim-A altında mekanik karar **ITERATE**, rev3.1'deki "ROLLBACK" retrospektif olarak hatalı.
  - §8 karar matrisi Rejim-A ve Rejim-B için ayrı tablolar.
  - §9 **Çok-Prompt Protokolü** eklendi: full 8-prompt ASAP1 eğitim/değerlendirme, MSE+rank loss, per-prompt QWK avg.
  - Eski ROLLBACK ablation chain'i (rb_lr_5e6, rb_lr_2e5, rb_dropout_02/03) rev3.2 eşiklerine göre **iptal** — rb_lr_5e6 sonucu (QWK 0.6782) Rejim-A'da ITERATE bandında.

---

*Onay sonrası bu belge `docs/` altında kalır; `.docx` karşılığı pandoc ile üretilir (`pandoc docs/ADR_001_rev3_addendum.md -o ADR_001_rev3_Addendum_v1.docx`).*
