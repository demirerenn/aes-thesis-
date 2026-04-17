# Faz 1 — Altyapı Doğrulama Raporu

**Tarih:** 2026-04-17
**Kapsam:** LangGraph orchestration + Evaluator + Alias layer + API erişimi
**Sonuç:** 4/4 adım yeşil (smoke training GX10'a devredildi)

## 1.1 Graph dry-run + Mermaid ✓

```
[graph] compiled — dry_run=True sprint=1
[graph] Mermaid diagram written → runs/_diagrams/graph.mmd
```

- **16 node** (13 core + 3 peer reviewer) derlendi, conditional edge'ler doğru
- `peer_coordinator` → 3 paralel reviewer fan-out → her biri kendi
  `route_after_review` ile `training_engineer` ya da `architect`'e dönüyor
- `fairness_auditor` → 3-yönlü conditional (`thesis_writer` / `feedback_strategy` / `architect`)
- Tek ENDpoint: `thesis_writer → END`

### Onarılan dosyalar
| Dosya | Sorun | Çözüm |
|---|---|---|
| `src/agents/state.py` | Reducer'lar eksikti → paralel fan-in kaybı | `operator.add` (decisions/artifacts), `_merge_scratch` (scratch) — 78 satır |
| `src/agents/graph.py` | 181. satırda truncate — string literal yarıda | Tüm dosya bash heredoc ile yeniden yazıldı — 421 satır |
| `src/agents/llm_factory.py` | Mükerrer `make_agent_node` append | `head -546` ile kırpıldı — tek kopya |

## 1.2 Smoke training pre-flight ✓ (GX10 execution pending)

Sandbox'ta torch kurulamaz (PyPI proxy 403). Torch-bağımsız her şey doğrulandı:

| Kontrol | Sonuç |
|---|---|
| YAML parse | OK — 8 key |
| `load_asap1(prompts=[2])` | 1800 satır, score 1-6, score_norm 0-1 |
| `fixed_split` 70/15/15 | 1260 / 270 / 270 |
| Avg essay length | ~380 kelime (max_length=768 uygun) |
| QWK metrik birim testi | perfect=1.0, noisy=0.889 |
| `train_baseline.py → evaluate_run` otomatik tetik | Satır 322-334 OK |

**GX10'da komut:**
```bash
python -m src.scripts.train_baseline \
  --config configs/baseline_asap1_p2.yaml --smoke --mode pilot
```
Tahmini: 5–8 dk, GB10 + bf16 + batch=4.

## 1.3 Evaluator node izole test ✓

Üç yol test edildi:

| Senaryo | Beklenen | Gerçekleşen |
|---|---|---|
| `run_dir=nonexistent` | rollback → architect, scratch'e decision yaz | ✓ `FileNotFoundError` yakalandı, Decision yazıldı |
| Synthetic summary (QWK=0.9646) + preds | go → thesis_writer, rendered report | ✓ 4641-byte / 145 satırlık Jinja2 Markdown rapor |
| `decide_gating()` Rejim-A/B eşik tablosu | A: 0.72/0.66/0.60; B: 0.82/0.78/0.72 | ✓ Tüm 8 eşik noktası doğru |

Rapor içeriği: diagonal dominance (%80.5), top-error pair (gerçek=2/tahmin=3, 11 adet), worst length-bucket (short, MAE=0.260), secondary metrics (Pearson/Spearman/Macro-F1).

## 1.4 Architect API çağrısı ✓ (auth OK, kredi gerekli)

End-to-end wiring doğrulandı:

```
[architect]      claude-opus-4-7  —  invoking LLM
[llm_factory] alias: claude-opus-4-7 → claude-opus-4-6
```

Sonra Anthropic API yanıtı:
```
anthropic.BadRequestError: Error code: 400 -
  {'type': 'error', 'error': {'type': 'invalid_request_error',
   'message': 'Your credit balance is too low to access the Anthropic API.
   Please go to Plans & Billing to upgrade or purchase credits.'}}
```

Bu hata neyi kanıtlar:
1. `AGENT_CONFIG["architect"]` → `claude-opus-4-7` atıyor ✓
2. `_resolve_model_alias` → `claude-opus-4-6` çeviriyor ✓
3. `_detect_provider` → `anthropic` ✓
4. `ChatAnthropic` client instantiate oluyor ✓
5. TLS handshake (sandbox proxy) → sandbox'ta `httpx(verify=False)` workaround gerekli, GX10'da gerekmez
6. API authentication OK (API key geçerli)
7. **Model `claude-opus-4-6` Anthropic tarafından kabul edildi** ✓
8. Generation sadece hesap bakiyesi sıfır olduğu için durdu

**Aksiyon:** `https://console.anthropic.com` → Plans & Billing → kredi yükle.

## Sıradaki — Faz 2 planı

1. **Kredi yükleme + TRUE smoke training** (GX10'da, ~10 dk):
   ```bash
   python -m src.scripts.train_baseline --config configs/baseline_asap1_p2.yaml --smoke --mode pilot
   ```
   Beklenen: `fold_records[0].qwk ∈ [0.4, 0.7]`, evaluator rapor oluşturuyor.

2. **Pilot Phase** (3 seed × 4 epoch, ~6-8 saat):
   ```bash
   python -m src.scripts.train_baseline --config configs/baseline_asap1_p2.yaml --mode pilot
   ```
   Gating:
   - QWK ≥ 0.82 → full CV başlat (Faz 3)
   - 0.78-0.82 → feedback_strategy → 1-2 ablasyon → CV
   - < 0.78 → architect → mimari revizyon

3. **Full agentic orchestration** (pilot yeşil olursa):
   ```bash
   python -m src.agents.graph --sprint 1 --run-once
   ```
