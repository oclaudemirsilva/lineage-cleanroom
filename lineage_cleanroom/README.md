# Lineage CleanRoom — Built with Claude: Life Sciences

> Projeto **à parte**, autocontido nesta pasta. Para o hackathon **Built with Claude: Life Sciences**
> (Anthropic + Gladstone Institutes + Cerebral Valley), virtual global, **7–13/jul/2026**,
> prêmio $100k em créditos.

**Lineage CleanRoom** é uma camada de **proveniência e anti-contaminação** para ML científico. Ela
responde a pergunta que todo revisor, clínico ou regulador realmente faz — **"dá pra confiar nesse
número?"** — checando as duas coisas que corrompem resultados sem fazer barulho (vazamento de dados e
proveniência de rótulo), e **assina** o veredito num manifesto à prova de adulteração (Ed25519).

O nome: *lineage* = **data lineage** (proveniência de dados) + **cell lineage** (linhagem celular);
*cleanroom* = o ambiente de **controle de contaminação** de todo laboratório. Juntos: um cleanroom para
a linhagem do seu dado.

---

## ✅ PROVA — roda, é testado, e tem CLI real

**Auditoria de um dataset real (o caso do pesquisador: "já separei treino/teste, está vazando?"):**
```
python -m lineage_cleanroom scan --data dataset.csv --label activity --split-col split \
    --group donor --provenance label_source --out ./audit
```
Saída real sobre um CSV de 2000 linhas / 64 features:
```
  Lineage CleanRoom -- CONTAMINATION [FAIL]
  CONTAMINATION DETECTED - LEAKAGE - 63.5% of test rows overlap training
  signed manifest verifies: True
  manifest -> ./audit/lineage_cleanroom.manifest.json
  report   -> ./audit/lineage_cleanroom.report.md
  (exit code 1 — amigável a CI)
```

**Demonstração "catch-the-leak" (antes/depois numa tela):** `python -m lineage_cleanroom demo`
```
  AUC (split ingênuo, VAZADO) ........ 0.9995   <- inflada
  AUC (split LIMPO, group-aware) ..... 0.6532   <- honesta
  Inflação por vazamento ............. 0.3463
  Manifesto assinado verifica ........ True
  Verifica APÓS adulterar 1 campo .... False    <- rejeita a fraude
```
O vazamento fez um modelo **medíocre (0.65)** parecer **quase perfeito (0.9995)**.

**Testes:** `python -m pytest lineage_cleanroom/test_cleanroom.py -q` → **14/14 verdes**.

---

## Os dois gates

1. **Gate de vazamento (features)** — pega linhas de teste que são duplicata exata de features do
   treino, e grupos (doador/batch/região) que cruzam a fronteira treino/teste. [`leakage.py`](leakage.py)
2. **Gate de proveniência (origem do rótulo)** — classifica a origem de cada rótulo (human / model /
   heuristic / unknown), exige que o gabarito seja **atestado por humano**, e pode proibir rótulos
   gerados por modelo no treino (**anti-autofagia**). [`provenance.py`](provenance.py)

Saída: **manifesto assinado Ed25519** (`*.manifest.json`) + relatório humano (`*.report.md`).

---

## Arquitetura (SOLID · baixo acoplamento · observabilidade)

Cada módulo tem uma responsabilidade; o core não conhece arquivo, classificador nem sink de telemetria
concretos — tudo por injeção de dependência (DIP). Novo modelo/split/fonte/sink = plugar, sem tocar o core.

| Módulo | Responsabilidade |
|---|---|
| [`ingest.py`](ingest.py) | Carrega dataset real de CSV (um arquivo c/ split, ou par treino/teste) → `SplitView`. |
| [`leakage.py`](leakage.py) | Detector de vazamento (feature-hash + group-aware). |
| [`provenance.py`](provenance.py) | Gate de proveniência de rótulo + política gold-humano / anti-autofagia. |
| [`splits.py`](splits.py) | Estratégias de split injetáveis (naive vaza / group-aware limpo). |
| [`manifest.py`](manifest.py) | Manifesto assinado Ed25519, tamper-evident. |
| [`report.py`](report.py) | Escreve `manifest.json` + `report.md` reprodutíveis. |
| [`telemetry.py`](telemetry.py) | Observabilidade por sink injetado (console/JSONL/fan-out), fail-soft. |
| [`pipeline.py`](pipeline.py) | Core: `audit_split` (produto) + `run_catch_the_leak` (demonstração). |
| [`cli.py`](cli.py) / [`__main__.py`](__main__.py) | CLI `scan` / `demo`, exit-code CI-friendly. |
| [`SKILL.md`](SKILL.md) | Empacotamento como Claude Code skill (Development Track). |
| [`datagen.py`](datagen.py) | Gerador sintético — **fixture de teste, não o produto**. |
| [`test_cleanroom.py`](test_cleanroom.py) | 14 testes (vazamento, proveniência, ingestão, audit, CLI). |

---

## O que está PRONTO vs o que FALTA

**Pronto (roda + testado):** ingestão real de CSV; gate de vazamento; gate de proveniência de rótulo
(gold-humano + anti-autofagia); auditoria `audit_split` sem treino; manifesto assinado + report em disco;
CLI `scan`/`demo` com exit-code; telemetria injetada; empacotamento como skill; 14 testes verdes.

**Falta (build da semana):** adapters de fonte além de CSV (parquet, matrizes single-cell, FASTA);
relatório visual de 1 tela (HTML); rodar sobre o **dataset real do hackathon**; *stretch:* interpretação
de rede PPI (Krogan) com cada aresta atestada+citada.

---

## Onde se aplica / relevância científica

**Áreas (mesma engenharia, várias fontes):** genômica/epigenômica (atividade regulatória de DNA,
sobreposição de sequência); telas CRISPR/single-cell do Marson (vazamento por doador/batch); redes PPI
do Krogan (treinar em arestas preditas = autofagia); imagem médica (vazamento por paciente);
clínico/GxP (proveniência assinada por resultado).

**Por que importa:** data leakage e treino em rótulos gerados por modelo são causas **documentadas e
crescentes** de resultados irreproduzíveis e retratações em biologia computacional. O Claude Science
registra o histórico de um artefato, mas **não impede** dado contaminado nem **certifica** a proveniência
do rótulo de forma tamper-evident. Lineage CleanRoom fecha essa lacuna — é a infraestrutura de
integridade de dados sob a qual qualquer métrica passa a ser confiável.

---

## A APLICAÇÃO (pronta para enviar — inglês)

**Project:** Lineage CleanRoom · **Track:** Development (demo on a Lab-Track dataset).
**One-liner:** A provenance and contamination firewall for scientific ML that plugs into Claude Science:
it detects data leakage, enforces human-attested ground truth, and emits a signed reproducibility
certificate.
**Problem:** train/test leakage and model-generated labels silently inflate metrics — a leading cause of
irreproducible results. Claude Science records history but does not prevent leakage or certify label
provenance.
**Solution:** a thin CLI + Claude Code skill that wraps any pipeline — detects leakage (feature-hash +
group-aware splits), audits label provenance (gold = human-attested only; anti-autophagy), and emits a
tamper-evident Ed25519 manifest.
**Demo:** on a DNA regulatory-activity dataset, a naive predictor shows AUC 0.9995; CleanRoom finds 63.5%
of test rows leaked; the honest clean AUC is 0.65; a signed manifest explains the leak.
**Why us:** months hardening exactly this (anti-contamination gate, human-attested canary, measurement
auditor, signed witness) inside a demanding video-ML corpus — we bring data-integrity engineering, not a
pretense of domain biology expertise.

---

## Links
- Evento: https://cerebralvalley.ai/e/built-with-claude-life-sciences
- Claude Science: https://www.anthropic.com/news/claude-science-ai-workbench
