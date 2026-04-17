"""Multi-provider LLM factory — tek yerden tüm agent'lar için ChatModel üretir.

Senin gösterdiğin basit yapıya benzer mantık:
    client = anthropic.Anthropic(api_key=...)
    response = client.messages.create(model=cfg["model"], system=cfg["system"], ...)

Ama bizim sistemde 3 sağlayıcı var (Anthropic, OpenAI, Google) ve LangGraph
ChatModel interface'i kullanıyoruz. Bu modül:
    1. Model string'inden sağlayıcıyı otomatik tespit eder
    2. Doğru ChatModel client'ı oluşturur (singleton — aynı model tekrar üretilmez)
    3. System prompt'u agent config'den alır
    4. .env'den API key'leri okur

Kullanım:
    from src.agents.llm_factory import get_chat_model, invoke_agent

    # Düşük seviye — ChatModel al
    llm = get_chat_model("claude-opus-4-7")
    response = llm.invoke([SystemMessage(...), HumanMessage(...)])

    # Yüksek seviye — agent adıyla çağır (senin run_agent fonksiyonuna benzer)
    result = invoke_agent("architect", "DeBERTa-v3-base mı large mı kullanmalıyız?")
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# .env dosyasını yükle (API key'ler burada)
load_dotenv()


# =====================================================================
# Agent Config — her agent'ın modeli, system prompt'u ve parametreleri
# =====================================================================
# Senin gösterdiğin AGENT_CONFIG yapısına benzer, ama daha detaylı.
# System prompt'lar Türkçe+İngilizce — agent'lar iki dilde çalışabilir.

AGENT_CONFIG: dict[str, dict[str, Any]] = {
    # ── Tier-1: Claude Opus 4.7 (stratejik muhakeme) ──────────────
    "orchestrator": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen AES (Automated Essay Scoring) tez projesinin orkestra yöneticisisin. "
            "Pipeline'ın mevcut durumunu analiz edip, hangi agent'ın ne yapması gerektiğine karar verirsin. "
            "Hedef: ASAP veri setinde QWK ≥ 0.80. Cihaz: ASUS Ascent GX10 (NVIDIA GB10, 128GB unified memory). "
            "Kararlarını gerekçelendir, her sprint sonunda durum özeti üret."
        ),
        "max_tokens": 2048,
        "temperature": 0.3,
    },
    "research": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen AES alanında literatür araştırma uzmanısın. "
            "Transformer tabanlı essay scoring modellerini (BERT, DeBERTa, RoBERTa, Longformer), "
            "kayıp fonksiyonlarını (MSE, CORAL, CORN, ordinal regression), ve değerlendirme metriklerini (QWK) "
            "derinlemesine bilirsin. R²BERT, PAES, T-SCORING gibi SOTA çalışmaları referans alırsın. "
            "Her bulguyu kaynak ve QWK değeriyle raporla."
        ),
        "max_tokens": 4096,
        "temperature": 0.2,
    },
    "data_analyst": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen ASAP 1.0 ve ASAP 2.0 veri setleri üzerinde EDA yapan uzman analistsin. "
            "Essay uzunluk dağılımı, prompt bazlı skor aralıkları, sınıf dengesizliği, "
            "outlier tespiti ve veri kalitesi analizi yaparsın. "
            "Bulgularını Architect agent'a stratejik içgörüler olarak iletirsin. "
            "Örn: 'P2 essay'leri ortalama 150 kelime → kısa metin modeli gerekebilir'."
        ),
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "architect": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen AES sistemi baş mimarısın. Kritik kararları sen alırsın: "
            "1) Backbone seçimi (DeBERTa-v3-base vs large vs RoBERTa vs Longformer) "
            "2) Head mimarisi (shared regression vs per-prompt head vs kova stratejisi) "
            "3) Kayıp fonksiyonu (MSE, MSE+rank, CORAL, CORN, label smoothing) "
            "4) Eğitim stratejisi (tek model vs prompt grupları vs ensemble) "
            "Kararlarını literatür, veri analizi ve önceki eğitim sonuçlarına dayandırırsın. "
            "Her kararı ADR (Architecture Decision Record) formatında belgelersin. "
            "Cihaz kısıtı: GX10 — 128GB unified memory, tek GPU."
        ),
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "training_engineer": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen ML eğitim mühendisisin. Architect'in kararlarını çalışan koda dönüştürürsün: "
            "- YAML config dosyaları (model, loss, optimizer, scheduler parametreleri) "
            "- Training script güncellemeleri (yeni loss, yeni head, data augmentation) "
            "- Hiperparametre seçimi (LR, batch size, epoch, warmup, gradient accumulation) "
            "GX10 kısıtlarını bilirsin: DeBERTa-v3-large batch 8'de ~40GB, base batch 16'da ~20GB. "
            "bf16 mixed precision kullanırsın. Kodun reproducible olmalı (seed, deterministic ops)."
        ),
        "max_tokens": 4096,
        "temperature": 0.2,
    },
    "feedback_strategy": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen eğitim sonuçlarını analiz eden stratejik danışmansın. "
            "Evaluator'dan gelen raporu (QWK, overfitting gap, per-prompt breakdown, confusion matrix) "
            "okur ve kök-neden analizi yaparsın: "
            "- Overfitting varsa: capacity mismatch mı, co-adaptation mı, data leakage mı? "
            "- Zayıf prompt'lar varsa: veri yetersizliği mi, skor aralığı sorunu mu, essay uzunluğu mu? "
            "Sonra somut ablasyon önerileri sunarsın (backbone değiştir, loss değiştir, head değiştir). "
            "Önerilerini öncelik sırasıyla ve beklenen etkisiyle birlikte raporla."
        ),
        "max_tokens": 4096,
        "temperature": 0.4,
    },
    "thesis_writer": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen akademik tez yazarısın. Türkçe akademik yazım kurallarına hakimsin. "
            "Eğitim sonuçlarını, ablasyon çalışmalarını ve mimari kararları "
            "tutarlı bir akademik anlatıya dönüştürürsün. "
            "Tablolar, grafikler ve istatistiksel analizlerle desteklersin. "
            "Her bölüm: giriş, yöntem, bulgular, tartışma formatında olmalı. "
            "Negatif sonuçları da (dropout başarısızlığı gibi) bilimsel değer olarak raporlarsın."
        ),
        "max_tokens": 8192,
        "temperature": 0.3,
    },
    "review_reproducibility": {
        "model": "claude-opus-4-7",
        "system": (
            "Sen bilimsel tekrarlanabilirlik (reproducibility) denetçisisin. "
            "Eğitim pipeline'ını şu açılardan denetlersin: "
            "- Seed sabitleme (Python, NumPy, PyTorch, CUDA) "
            "- Veri sızıntısı kontrolü (train/val/test ayrımı, prompt-level split) "
            "- İstatistiksel geçerlilik (çoklu seed, CI aralıkları, p-değerleri) "
            "- Checkpoint ve artifact versiyonlama "
            "Sorun bulursan REJECT + somut düzeltme önerisi ver."
        ),
        "max_tokens": 2048,
        "temperature": 0.1,
    },

    # ── Tier-2: Claude Sonnet 4.6 (yapılandırılmış analiz) ────────
    "evaluator": {
        "model": "claude-sonnet-4-6",
        "system": (
            "Sen eğitim sonuçlarını değerlendiren ve gating kararı veren agent'sın. "
            "NOT: Bu agent büyük ölçüde deterministik — evaluator.py Python kodu "
            "QWK eşiklerine göre GO/ITERATE/REVISE/ROLLBACK kararını verir. "
            "LLM olarak senin görevin sadece raporu cilalamak ve Türkçe özet eklemek."
        ),
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "fairness_auditor": {
        "model": "claude-sonnet-4-6",
        "system": (
            "Sen ASAP 2.0 / PERSUADE veri setinde demografik adalet denetçisisin. "
            "Grup bazlı QWK farklılıkları, DIF (Differential Item Functioning), "
            "ve skor dağılımı eşitliğini izlersin. Eşitsizlik tespit edersen alarm ver."
        ),
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "devops": {
        "model": "claude-sonnet-4-6",
        "system": (
            "Sen GX10 altyapı mühendisisin. Docker container yönetimi, "
            "NGC base image güncellemeleri, CUDA uyumluluğu, ARM64 (aarch64) "
            "özel sorunları ve volume mount'ları senin sorumluluğunda. "
            "Cihaz: NVIDIA GB10 Grace Blackwell, unified memory 128GB."
        ),
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "ops_monitor": {
        "model": "claude-sonnet-4-6",
        "system": (
            "Sen operasyonel izleme agent'ısın. Epoch bazlı GPU bellek kullanımı, "
            "eğitim süresi trendi, loss eğrisi anomalileri ve early stopping "
            "sinyallerini izlersin. Overfitting erken tespiti kritik görevin."
        ),
        "max_tokens": 1024,
        "temperature": 0.1,
    },
    "peer_coordinator": {
        "model": "claude-sonnet-4-6",
        "system": (
            "Sen peer review sürecini koordine edersin. 3 reviewer'dan gelen "
            "raporları toplar, çelişkileri tespit eder, konsensüs oluşturursin. "
            "3/3 onay → Training Engineer'a ilerle. Aksi halde → Architect'e geri gönder."
        ),
        "max_tokens": 2048,
        "temperature": 0.2,
    },

    # ── Tier-3: Cross-provider (kognitif çeşitlilik) ──────────────
    "code_reviewer": {
        "model": "gpt-5.3-codex",
        "system": (
            "You are a senior ML code reviewer. Review training scripts, "
            "data pipelines, and model implementations for: "
            "- Correctness (loss computation, gradient flow, metric calculation) "
            "- Performance (unnecessary copies, memory leaks, inefficient tokenization) "
            "- Reproducibility (seed handling, deterministic ops) "
            "- Best practices (type hints, logging, error handling) "
            "Flag issues as CRITICAL / WARNING / INFO with fix suggestions."
        ),
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "review_ml_logic": {
        "model": "gemini-3.1-pro",
        "system": (
            "You are an ML methodology reviewer for an Automated Essay Scoring thesis. "
            "Evaluate: model architecture choices, loss function suitability for ordinal targets, "
            "evaluation protocol (per-prompt QWK averaging), cross-validation strategy, "
            "and statistical validity of reported results. "
            "Compare against SOTA: R²BERT (0.794 avg QWK), PAES, Taghipour & Ng (0.761). "
            "Be rigorous — this is for a thesis defense."
        ),
        "max_tokens": 4096,
        "temperature": 0.2,
    },
    "review_performance": {
        "model": "gpt-5.4-pro",
        "system": (
            "You are a performance metrics auditor for an AES system. "
            "Validate: QWK calculation correctness, per-prompt vs overall aggregation, "
            "confidence interval computation, statistical significance of improvements, "
            "and comparison fairness against literature baselines. "
            "Check for p-hacking, selective reporting, and metric gaming."
        ),
        "max_tokens": 2048,
        "temperature": 0.1,
    },
}


# =====================================================================
# Provider detection & ChatModel factory
# =====================================================================

def _detect_provider(model_id: str) -> str:
    """Model string'inden sağlayıcıyı tespit et."""
    if model_id.startswith("claude-"):
        return "anthropic"
    elif model_id.startswith("gpt-"):
        return "openai"
    elif model_id.startswith("gemini-"):
        return "google"
    else:
        raise ValueError(
            f"Bilinmeyen model sağlayıcısı: {model_id}. "
            f"Desteklenen prefix'ler: claude-*, gpt-*, gemini-*"
        )


@lru_cache(maxsize=32)
def get_chat_model(model_id: str, temperature: float = 0.3) -> BaseChatModel:
    """Model ID'den uygun ChatModel client'ı oluştur (singleton / cached).

    Senin gösterdiğin yapıdaki şuna karşılık gelir:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    Ama burada 3 sağlayıcı otomatik yönetiliyor.
    """
    provider = _detect_provider(model_id)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY bulunamadı. .env dosyasına ekleyin: "
                "https://console.anthropic.com → API Keys"
            )
        return ChatAnthropic(
            model=model_id,
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY bulunamadı. .env dosyasına ekleyin: "
                "https://platform.openai.com/api-keys"
            )
        return ChatOpenAI(
            model=model_id,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY bulunamadı. .env dosyasına ekleyin: "
                "https://aistudio.google.com/apikey"
            )
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=4096,
        )

    raise ValueError(f"Desteklenmeyen provider: {provider}")


# =====================================================================
# Yüksek seviye invoke — senin run_agent() fonksiyonuna karşılık gelir
# =====================================================================

def invoke_agent(agent_name: str, user_message: str, **kwargs) -> str:
    """Agent adıyla LLM çağrısı yap.

    Senin örneğindeki şuna karşılık gelir:
        plan = run_agent("planner", "Tez konum için araştırma planı çıkar")

    Ama burada:
        - Model otomatik seçilir (AGENT_CONFIG'den)
        - System prompt otomatik eklenir
        - 3 farklı sağlayıcı desteklenir
        - .env override'ları dikkate alınır (AES_LLM_ARCHITECT=... gibi)
    """
    if agent_name not in AGENT_CONFIG:
        raise KeyError(
            f"Bilinmeyen agent: '{agent_name}'. "
            f"Mevcut agent'lar: {list(AGENT_CONFIG.keys())}"
        )

    cfg = AGENT_CONFIG[agent_name]

    # .env override kontrolü (resolve_llm mantığı)
    env_key = f"AES_LLM_{agent_name.upper()}"
    model_id = os.getenv(env_key) or cfg["model"]

    temperature = kwargs.get("temperature", cfg.get("temperature", 0.3))
    max_tokens = kwargs.get("max_tokens", cfg.get("max_tokens", 4096))

    llm = get_chat_model(model_id, temperature=temperature)

    messages = [
        SystemMessage(content=cfg["system"]),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return response.content


# =====================================================================
# LangGraph node factory — graph.py'deki _stub() yerine kullanılacak
# =====================================================================

def make_agent_node(agent_name: str):
    """LangGraph node fonksiyonu üret — graph.py'deki _stub()'ın gerçek hali.

    Bu fonksiyon AESState alır, agent'ı çağırır, sonucu state'e yazar.
    """
    from .state import AESState

    def node(state: AESState) -> AESState:
        cfg = AGENT_CONFIG.get(agent_name, {})
        model_id = os.getenv(f"AES_LLM_{agent_name.upper()}") or cfg.get("model", "unknown")

        # State'ten ilgili context'i hazırla
        context_parts = []
        if state.get("decisions"):
            last_decisions = state["decisions"][-3:]  # son 3 karar
            for d in last_decisions:
                context_parts.append(f"[{d['agent']}] {d['decision']}: {d['rationale']}")

        if state.get("best_qwk"):
            context_parts.append(f"Mevcut en iyi QWK: {state['best_qwk']:.4f}")

        if state.get("current_run_name"):
            context_parts.append(f"Aktif run: {state['current_run_name']}")

        context = "\n".join(context_parts) if context_parts else "Henüz context yok — ilk sprint."

        print(f"[{agent_name:>22}] {model_id:>20}  —  invoking LLM")

        try:
            result = invoke_agent(agent_name, context)

            # Sonucu state'e ekle
            from langchain_core.messages import AIMessage
            state.setdefault("messages", []).append(
                AIMessage(content=f"[{agent_name}] {result[:500]}")  # truncate for state size
            )

            # Visit tracking
            state.setdefault("scratch", {}).setdefault("visits", {}).setdefault(agent_name, 0)
            state["scratch"]["visits"][agent_name] += 1
            state["scratch"][f"{agent_name}_last_output"] = result

            print(f"[{agent_name:>22}] ✓ completed ({len(result)} chars)")

        except EnvironmentError as e:
            # API key eksik — stub moduna düş
            print(f"[{agent_name:>22}] ⚠ API key eksik, stub modunda: {e}")
            state.setdefault("scratch", {}).setdefault("visits", {}).setdefault(agent_name, 0)
            state["scratch"]["visits"][agent_name] += 1

        return state

    return node
