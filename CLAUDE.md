# CLAUDE.md

> Ce fichier est lu automatiquement par Claude Code (et utile à coller en début de
> conversation Claude.ai) pour qu'il comprenne le projet sans réexplications.
> Il combine : contexte du projet, conventions de code, préférences de
> collaboration, et roadmap.

---

## 1. Project context

**Project name**: Scientific Watch Agent
**Type**: Master's thesis (PFE) - 3 to 4 months
**Field**: Computer Science / AI / Data Science
**Author**: [À compléter par l'étudiant]

### Vision en une phrase

Un système multi-agents qui, à partir d'un topic de recherche, produit
automatiquement une veille scientifique complète : articles filtrés par qualité,
résumés structurés, synthèse globale, identification de tendances et
perspectives de recherche.

### Utilisateur cible

**Chercheurs académiques en début de thèse** qui doivent rapidement comprendre
un domaine inconnu : état de l'art, approches dominantes, datasets de référence,
gaps de recherche, directions émergentes.

### Objectif principal

Construire un **système réel, utilisable et généralisable** (pas un proof-of-
concept) qui sera validé sur **plusieurs domaines** pour démontrer sa
généralisation cross-domaine.

### Critères de succès du mémoire (équilibre 3 axes)

1. **Engineering** : code propre, tests, documentation, architecture maintenable
2. **Recherche** : expérimentations rigoureuses avec métriques quantitatives
3. **Produit** : interface utilisable, démo convaincante

---

## 2. Scientific contribution

### Contribution la plus originale : généralisation cross-domaine (méta-learning)

Le cœur scientifique du mémoire est de démontrer qu'un agent de veille
scientifique peut **généraliser à de nouveaux domaines de recherche** sans
re-entraînement spécifique, grâce à des techniques de méta-learning appliquées
aux configurations du système (pondérations de scoring, prompts, choix de
modèles).

L'idée centrale : au lieu d'apprendre des paramètres optimaux **pour un
domaine** (fake news detection), on apprend une **fonction qui prédit les bons
paramètres à partir de meta-features du domaine** (vocabulaire technique,
maturité du champ, distribution des citations, etc.).

### Niveaux de généralisation à étudier

1. **Niveau 1 - Transfer direct** : utiliser les pondérations apprises sur
   Domaine A telles quelles sur Domaine B. Mesure : performance relative.
2. **Niveau 2 - Fine-tuning rapide** : ajuster les pondérations sur Domaine B
   avec très peu de samples (few-shot). Mesure : nb de samples nécessaires.
3. **Niveau 3 - Méta-apprentissage** : apprendre une fonction
   `meta_features(domain) → optimal_weights` à partir de plusieurs domaines.

### Contributions secondaires (qui servent la contribution principale)

- **Multi-agents (V2 + V3)** : architecture qui permet d'identifier *où* la
  généralisation casse (quel agent dégrade le plus en cross-domaine ?)
- **AutoML pour scoring scientifique** : Optuna apprend les pondérations
  optimales par domaine (base pour le méta-learning)
- **Pattern Reflexion** : le Critic peut détecter les hallucinations
  cross-domaine
- **Comparaison Claude vs GPT** : quel modèle généralise le mieux ? Hypothèse
  intéressante car les deux ont des distributions d'entraînement différentes
- **Contrat de communication formalisé** (Pydantic v2)

### Hypothèses de recherche — statut (juillet 2026)

- **H1 - AutoML par domaine** ✅ **CONFIRMÉE** : Optuna bat les poids experts sur
  les 19 domaines évalués (amélioration moyenne +68.7% NDCG@15 vs Default).
  Voir Section 14 pour les résultats détaillés par domaine.
- **H2 - Transfer direct dégrade de façon prévisible** : Transfer Direct = +7.7%
  vs Default (mean NDCG@15 0.247 vs 0.229). La dégradation est corrélée à la
  distance inter-domaine via les meta-features. Partiellement validée.
- **H3 - Le méta-learning bat le transfer direct** ✅ **CONFIRMÉE** :
  Borda Count (ensemble kNN(1)+kNN(2)+kNN(3)+BayesRidge) bat Transfer Direct
  sur **15/19 domaines = 79% ≥ 75% seuil** (Wilcoxon W⁺=149, p=0.0145, r_rb=+0.57, large effect).
  Mean NDCG@15 Borda = 0.282 vs Transfer 0.247 (+14.2%).
- **H4 - Architecture multi-agents identifie les goulots** : 🔜 Phase 7 (Best-of-N)
- **H5 - Claude et GPT ont des profils différents** : 🔜 Phase 7 (comparaison A/B)

---

## 3. Architecture overview

### Roadmap par phases

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ done | Project setup |
| 1 | ✅ done | OpenAlex source wrapper |
| 2 | ✅ done | Quality scoring features (venue, authors, impact, relevance) |
| 3 | ✅ done | AutoML scorer with Optuna (6 features, 5 domains evaluated) |
| 4 | ✅ done | LLM layer (Claude + OpenAI + Groq + Nebius) + LLMClient Protocol + factory |
| 5 | ✅ done | LangGraph orchestration (V2): ReAct QueryExpander + tous les agents |
| 6 | ✅ done | Reflexion sur Synthesizer + ReAct sur QueryExpander + ProTeGi prompt opt. |
| 7 | 🔜 todo | V3 concurrent agents (Best-of-N Claude vs GPT, with Judge) |
| 8 | ✅ done | Méta-learning : meta_features(domain) → optimal_weights (H3 confirmed) |

### Pipeline réel (V2 + Reflexion + ReAct)

```
Topic
  └─► QueryExpander (ReAct loop, max 4 iter)
           └─► Searcher (multi-query fetch + dedup)
                    └─► QualityCritic (6-feature scoring, Optuna weights)
                             └─► Summarizer (structured | narrative mode)
                                      └─► Synthesizer ◄──────────────────┐
                                               └─► Critic (Reflexion) ────┘
                                                    │ needs_revision=False
                                                    └─► TrendAnalyst → END
```

### Stack technique

- **Language**: Python 3.10+
- **Validation**: Pydantic v2 (schemas Pydantic = communication contract)
- **Sources**: OpenAlex API (primary)
- **LLMs**: 4 providers — Claude (Anthropic), GPT (OpenAI), Llama/Qwen3 (Groq), DeepSeek/Qwen3 (Nebius)
- **Orchestration**: LangGraph (agents = nodes, WatchState = TypedDict partagé)
- **Patterns agents**: ReAct (QueryExpander), Reflexion (Synthesizer+Critic)
- **AutoML scoring**: Optuna TPE, 6 features, 5 domaines évalués
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (relevance V2)
- **Prompt optimization**: ProTeGi (text-gradient-based, src/optimization/)
- **Tests**: pytest, 266 tests, mocks pour tous les appels réseau/LLM

### Structure du projet (état réel juin 2026)

```
scientific-watch-agent/
├── src/
│   ├── config.py                    # Config centralisée (TASK_MODELS, weights, etc.)
│   ├── schemas.py                   # Pydantic v2 — THE communication contract
│   ├── sources/
│   │   └── openalex.py              # ✅ OpenAlex client
│   ├── features/                    # ✅ 6 scoring features
│   │   ├── venue_score.py           # quartile Q1-Q4
│   │   ├── authors_score.py         # max h-index (log)
│   │   ├── impact_score.py          # citations/year (log)
│   │   ├── relevance_score.py       # V1 keyword + bigrams
│   │   ├── relevance_score_v2.py    # V2 semantic embeddings (MiniLM)
│   │   ├── velocity_score.py        # citations momentum (linear)
│   │   └── recency_score.py         # publication freshness (exp decay)
│   ├── scoring/                     # ✅
│   │   ├── quality_scorer.py        # 6-feature orchestrator
│   │   ├── automl_scorer.py         # Optuna TPE, _FEATURE_VERSION="f6"
│   │   └── metrics.py               # NDCG@k, P@k, Recall@k, MAP
│   ├── llm/                         # ✅ Phase 4 — 4 providers
│   │   ├── base.py                  # LLMClient Protocol + LLMResponse + cost
│   │   ├── claude_client.py         # Anthropic (tool_use structured output)
│   │   ├── openai_client.py         # OpenAI (response_format structured output)
│   │   ├── groq_client.py           # Groq (Llama, Qwen3 — ultra-fast)
│   │   ├── nebius_client.py         # Nebius AI Studio (DeepSeek, Qwen3)
│   │   └── factory.py               # get_llm_for_task(task) → LLMClient
│   ├── agents/                      # ✅ Phase 5+6 — pipeline complet
│   │   ├── state.py                 # WatchState TypedDict + RunConfig
│   │   ├── base.py                  # start_log / finish_log helpers
│   │   ├── graph.py                 # LangGraph pipeline + route_after_critic
│   │   ├── query_expander.py        # ReAct loop (max 4 iter, OpenAlex probes)
│   │   ├── searcher.py              # multi-query fetch + dedup
│   │   ├── quality_critic.py        # 6-feature scoring + Optuna weights loader
│   │   ├── summarizer.py            # structured mode + narrative mode (prose)
│   │   ├── synthesizer.py           # 1st run + Reflexion revision
│   │   ├── critic.py                # Reflexion: évalue sur 4 axes, force-approve
│   │   └── trend_analyst.py         # trends/gaps/perspectives + retry auto
│   └── optimization/                # ✅ ProTeGi
│       ├── protegi_optimizer.py     # text-gradient prompt search
│       └── summary_metrics.py       # ROUGE, BERTScore
├── tests/                           # 266 tests passing (pytest + mocks)
├── data/
│   ├── oracle/                      # 19 domaines : gold_articles + gold_relevance
│   │   └── domains_config.json      # config des 19 domaines (topic, from_year)
│   ├── weights/                     # 19 fichiers JSON de poids Optuna appris
│   ├── metalearning/
│   │   ├── meta_features.json       # meta-features extraites (19 domaines)
│   │   └── phase8c_results.json     # résultats LOO-CV (15 méthodes × 19 domaines)
│   └── plots/                       # graphes Phase 8
│       ├── diag_method_wins.png     # tableau win/loss 12 méthodes × 19 domaines
│       └── phase8d_wilcoxon.png     # p-values + effect sizes Wilcoxon
├── src/
│   └── metalearning/                # ✅ Phase 8
│       ├── meta_features.py         # DomainMetaFeatures dataclass + extraction
│       └── meta_learner.py          # MetaLearner : kNN(1-3), Ridge, BayesRidge,
│                                    #   GPR, SoftVote, HardVote, Borda, RRF,
│                                    #   Ensemble (kNN+Ridge), Ensemble (kNN+Bayes)
├── phase8a_8b.py                    # extraction meta-features + corrélations Pearson
├── phase8c.py                       # LOO-CV : 15 méthodes × 19 domaines → NDCG@15
├── phase8d_wilcoxon.py              # test Wilcoxon signé (H3) + graphe p-values
├── compare_v1_v2.py                 # Optuna par domaine (V1.5 keyword vs V2 embeddings)
├── diag_method_wins.py              # tableau PNG win/loss par domaine
├── demo_phase1.py                   # OpenAlex search
├── demo_phase2.py                   # search + score + filter
├── demo_phase3.py                   # AutoML Optuna (--domain ou --synthetic)
├── demo_protegi.py                  # ProTeGi prompt optimization
├── promote_oracle_grade2.py         # fix oracles -- options: --select INDICES --force
├── visualize_protegi.py             # graphes résultats ProTeGi
├── diag_oracle.py                   # diagnostic oracle (IDs, grades)
├── requirements.txt
└── .env                             # API keys (NOT committed)
```

---

## 4. Communication contract (very important)

Le contrat de communication entre agents est **central** au projet. Il est
défini dans `src/schemas.py` avec Pydantic v2.

### Schémas principaux

- `Article` : entité centrale (id, title, abstract, authors, citations, venue)
- `Author` : avec h_index, citation_count, affiliation
- `QualityScore` : output du QualityCritic — **6 sub-scores** (venue, authors, impact, velocity, recency, relevance) + final + weights_used
- `ArticleSummary` : output du Summarizer — mode structuré (problem, method, dataset, results, limits, contribution)
- `NarrativeSummary` : output du Summarizer — mode narrative (prose paragraph, 150-250 mots)
- `Synthesis` : output du Synthesizer (overview, main_approaches, common_datasets, key_findings)
- `TrendAnalysis` : output du TrendAnalyst (trends avec maturity, gaps avec importance, future_perspectives)
- `CriticFeedback` : pour le pattern Reflexion (overall_quality, issues, suggestions, needs_revision)
- `ReActThoughtAction` : output du QueryExpander (thought, action, search_query, stop_reason)
- `AgentLog` : trace d'exécution (tokens_used, api_calls, cost_usd, duration)

### Règles d'ownership du shared state

- **Single Writer / Multiple Readers** : chaque section du state n'est écrite
  que par UN agent, lue par tous les autres
- **Lecture libre** sur l'ensemble du state pour tout agent
- **Validation Pydantic** à chaque transition (lenient en V1, strict en V2)

### Mécanisme de communication

- LangGraph **Shared State** (TypedDict partagé)
- PAS de message passing direct entre agents (pour le moment)
- Migration possible vers blackboard pattern en V3+ si besoin

---

## 5. Multi-LLM strategy

### Mapping tâche → modèle (production actuelle)

Voir `TASK_MODELS` dans `src/config.py` :

| Tâche | Provider | Modèle | Pourquoi |
|---|---|---|---|
| query_expansion | openai | gpt-4o-mini | Rapide, peu cher |
| summarize | nebius | Qwen3-30B-A3B-Instruct | Répétitif, cheap ($0.14/M) |
| synthesize | anthropic | claude-sonnet-4-6 | Grand contexte (200K), qualité |
| trend_analysis | anthropic | claude-sonnet-4-6 | Raisonnement complexe |
| critic | nebius | DeepSeek-V3.2 | Cheap + cohérent pour Reflexion |
| judge | nebius | DeepSeek-V3.2 | Même modèle = cohérence évaluation |

### 4 providers disponibles

| Provider | Client | Modèles utilisés | Structured output |
|---|---|---|---|
| Anthropic | `claude_client.py` | claude-haiku-4-5, sonnet-4-6, opus-4-7 | tool_use forcé |
| OpenAI | `openai_client.py` | gpt-4o-mini, gpt-4o | response_format JSON |
| Groq | `groq_client.py` | llama-3.3-70b, llama-3.1-8b | response_format JSON |
| Nebius | `nebius_client.py` | DeepSeek-V3.2, Qwen3-30B, Qwen3-235B | response_format JSON |

Groq et Nebius utilisent l'API compatible OpenAI — `NebiusClient` et `GroqClient`
héritent d'`OpenAIClient` avec un `base_url` différent.

### Trois modes d'utilisation

1. **Production** : spécialisation par tâche (mapping TASK_MODELS)
2. **Benchmark** : A/B testing sur paires définies dans `BENCHMARK_PAIRS`
3. **ProTeGi** : optimisation automatique des prompts par text-gradient search

### Optimisations de coût (actives)

- **Prompt caching Anthropic** sur system prompts longs (90% off inputs cachés)
- **Nebius/Groq** pour tâches répétitives : 10-20x moins cher que Claude/GPT
- **Tracker tokens + coût** dans `AgentLog.cost_usd` à chaque appel
- Coût production single-run : ~$0.05 avec config Nebius actuelle

---

## 6. Coding conventions

### Langue dans le code

- **Code, docstrings, comments, nom de variables** : English
- **Discussions, explications dans le chat** : Français
- **Messages d'erreur user-facing** : English (international standard)
- **README et docs externes** : English

### Style Python

- Python 3.10+ (pour syntaxes modernes : `X | Y`, `list[X]`)
- Type hints partout (incluant `from __future__ import annotations`)
- Pydantic v2 pour toute donnée structurée traversant des frontières
- Logging avec `logging` module (pas de `print` dans le code de prod)
- Constantes en `UPPER_SNAKE_CASE`, fonctions en `snake_case`

### Structure des modules

- Un module = une responsabilité claire
- Pas de classes "god" - préférer plusieurs petites classes/fonctions
- Configuration centralisée dans `src/config.py` (pas de hardcoding)
- Fonctions pures quand possible (testables sans mocks)

### Tests

- pytest avec mocks pour tout appel réseau et LLM
- Couverture : tous les cas limites (empty input, None, valeurs extrêmes)
- Tests paramétrés pour les cas similaires
- Naming : `test_<fonction>_<scenario>` ou classe `TestX::test_y`
- **266 tests passants** (état Phase 6)

### Gestion d'erreurs

- **V1 : mode souple** - try/except global, log les erreurs, continue
- **V2 : mode strict** - validation stricte, raise sur erreur
- Retry avec backoff exponentiel sur erreurs réseau (tenacity)
- Pas de silencing d'erreur sans log

---

## 7. Environment specifics

### Plateforme : Windows natif (PowerShell)

**Important pour les commandes** :
- Utiliser `;` pour chainer commandes en PowerShell (pas `&&`)
- Activation venv : `.\venv\Scripts\Activate.ps1` (pas `source venv/bin/activate`)
- Variables d'environnement : `$env:OPENALEX_EMAIL = "..."` ou via `.env`
- Path separators : Python gère les `/` correctement, mais PowerShell utilise `\`

### Setup standard

```powershell
# Création venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install
pip install -r requirements.txt

# Config
Copy-Item .env.example .env
# Édite .env

# Run tests
pytest tests/ -v

# Run demo
python demo_phase2.py "fake news detection" 30 10
```

---

## 8. Collaboration preferences

### Mon style préféré

1. **Pédagogique** : explique le pourquoi, pas juste le comment
2. **Bilingue** : code et docstrings en anglais, discussions en français
3. **Options multiples** : quand il y a plusieurs choix techniques, propose
   2-4 options avec pros/cons, je décide
4. **Structuré** : utilise les outils interactifs (questions à choix multiples)
   pour clarifier mes besoins
5. **Code testable** : toujours livrer code + tests qui passent

### Comment Claude doit prendre des décisions

| Type de décision | Approche |
|---|---|
| Architecturale (nouveau pattern, choix de lib) | Présenter options, demander choix |
| Implémentation détaillée (style, helpers) | Décider et expliquer brièvement |
| Stylistique (naming, formatting) | Suivre les conventions ci-dessus, pas demander |
| Coût/budget API | M'avertir si > 1$ par run prévu |
| Breaking change dans l'architecture | Toujours demander avant |

### Niveau d'aide attendu par sujet

- **Code Python général** : autonome
- **LangGraph** : ✅ maîtrisé (pipeline implémenté, conditional edges compris)
- **AutoML/Optuna** : ✅ maîtrisé (6 features, 5 domaines, TPE compris)
- **Reflexion pattern** : ✅ maîtrisé (Critic + Synthesizer revision implémentés)
- **ReAct pattern** : ✅ maîtrisé (QueryExpander implémenté)
- **ProTeGi** : ✅ maîtrisé (nombreuses expériences menées)
- **Multi-agents (Best-of-N, Debate)** : 🔜 prochaine phase (Phase 7)
- **Méta-learning** : ✅ maîtrisé (H3 confirmée, Borda 15/19, Wilcoxon p=0.0145)
- **Pydantic v2** : autonome
- **Pytest/mocks** : autonome
- **APIs Claude/OpenAI/Groq/Nebius** : autonome (4 providers intégrés)

---

## 9. Workflow recommendations

### Boucle de développement par phase

1. **Discussion d'architecture** (questions à choix multiples)
2. **Spécification du contrat** (schemas Pydantic)
3. **Implémentation TDD-light** (tests + code en parallèle)
4. **Validation end-to-end** (demo script)
5. **Documentation et update CLAUDE.md** (roadmap, status)

### Avant chaque grosse session

- Vérifier que les tests Phase précédente passent toujours
- Clarifier le scope de la session avec questions interactives
- Éviter les sessions de plus de 2 phases en une fois (digestible)

### Après chaque grosse session

- Tests passants + demo runs
- README mis à jour si nouveau script ou commande
- CLAUDE.md mis à jour si nouveau pattern ou décision architecturale
- Zip téléchargeable pour sauvegarde

---

## 10. Open questions / decisions pending

### Décisions prises (archivées)

- [x] Domaines de validation : 5 oracles construits (fake_news, federated, GNN, XAI, medical)
- [x] Embeddings V2 : all-MiniLM-L6-v2 local (pas Voyage AI — évite coût API)
- [x] Providers LLM : Claude + OpenAI + Groq + Nebius (4 implémentés)
- [x] Patterns agents : ReAct (QueryExpander) + Reflexion (Synthesizer+Critic)

### Décisions prises (Phase 8)

- [x] **Méta-learning architecture** : Borda Count (ensemble kNN(1-3)+BayesRidge) → méthode gagnante
- [x] **Meta-features sélectionnées** : 7 features avec |r| > 0.40 (voir Section 16)
- [x] **19 domaines d'évaluation** : 16 AI + 3 hors-AI (comp. bio., medical NLP, quant. finance)
- [x] **Oracle repair** : `promote_oracle_grade2.py --select INDICES --force` pour sélection manuelle

### Décisions encore ouvertes

- [ ] **Format rapport final** : est-ce que le pipeline génère un PDF / HTML exportable ?
- [ ] **Interface utilisateur finale** : CLI seul ou Streamlit ?
- [ ] **Phase 7 (Best-of-N)** : Best-of-N Claude vs GPT avec Judge (H4, H5)
- [ ] **Wilcoxon à α=0.10** : En+R (p=0.077), RRF (p=0.078), BayesRidge (p=0.095) deviennent significatifs — à mentionner dans le mémoire ?
- [ ] **n=19 → n≥30** : augmenter le nombre de domaines pour plus de puissance statistique ?

---

---

## 11. Thesis writing in parallel

L'utilisateur veut **écrire le mémoire en parallèle du code**. C'est la bonne
approche pour un PFE car elle :
- Évite la "panique de fin de PFE" (1 mois pour tout écrire)
- Force à clarifier les choix techniques au moment où on les prend
- Permet d'identifier les expériences manquantes plus tôt

### Stratégie : écrire chapitre par chapitre, en suivant les phases

| Chapitre du mémoire | Phase de code | Statut rédaction |
|---|---|---|
| 1. Introduction | Phase 0-1 | À rédiger |
| 2. État de l'art | Phase 0 (biblio) | À rédiger |
| 3. Méthodologie générale | Phase 1-2 (sources, scoring) | À rédiger |
| 4. AutoML pour scoring | Phase 3 ✅ | **Peut être rédigé maintenant** |
| 5. Architecture multi-agents | Phase 4-5 ✅ | **Peut être rédigé maintenant** |
| 6. Patterns avancés (Reflexion, ReAct, ProTeGi) | Phase 6 ✅ | **Peut être rédigé maintenant** |
| 7. Mode concurrent V3 (Best-of-N) | Phase 7 🔜 | Quand Phase 7 termine |
| 8. Méta-learning cross-domaine | Phase 8 ✅ | **Peut être rédigé maintenant** |
| 9. Résultats expérimentaux | Phase 8 | Itératif |
| 10. Discussion + Conclusion | Tout | À la fin |

### Comment Claude peut aider sur la rédaction

À chaque fin de phase, Claude peut :
1. **Synthétiser les choix techniques** dans une section "Methodology"
2. **Documenter les expériences** menées avec leurs résultats
3. **Rédiger les justifications scientifiques** des décisions de design
4. **Suggérer des références bibliographiques** pertinentes
5. **Relire et reformuler** des sections rédigées par l'étudiant

### Format et conventions de rédaction

- **Langue** : français (mémoire de PFE en France/zone francophone, à
  confirmer)
- **Style** : académique mais accessible, voix neutre ("nous proposons", "il
  est observé que"), phrases courtes
- **Citations** : style APA ou IEEE (à choisir tôt, souvent imposé par l'école)
- **Figures** : numérotées, avec captions explicites, référencées dans le texte
- **Tables** : idem, alignement professionnel
- **Code dans le mémoire** : extraits ciblés (5-15 lignes max), pas de blocs
  géants, expliqués dans le texte

### Outils recommandés pour la rédaction

- **LaTeX (Overleaf)** : standard académique, gestion auto des références
- **Markdown + Pandoc** : alternative plus simple, conversion facile en PDF
- **Word/LibreOffice** : si imposé par l'école, sinon éviter (gestion des
  références plus pénible)

### Demander à Claude de rédiger

Quand l'utilisateur demande "rédige la section X du chapitre Y", Claude doit :
1. Demander le ton attendu (formel, semi-formel)
2. Demander la longueur cible (en pages ou mots)
3. Demander si des éléments précis doivent absolument figurer
4. Produire un brouillon en français académique
5. Suggérer 2-3 améliorations possibles après le draft

---

## 12. References utiles

### Documentation des outils

- **OpenAlex API** : https://docs.openalex.org/
- **Pydantic v2** : https://docs.pydantic.dev/latest/
- **LangGraph** : https://langchain-ai.github.io/langgraph/
- **Optuna** : https://optuna.readthedocs.io/
- **Claude API** : https://docs.claude.com/
- **OpenAI API** : https://platform.openai.com/docs

### Papers de référence (pour le mémoire)

- **Reflexion** : Shinn et al., "Reflexion: Language Agents with Verbal
  Reinforcement Learning" (2023)
- **Multi-agent Debate** : Du et al., "Improving Factuality and Reasoning in
  Language Models through Multiagent Debate" (MIT, 2023)
- **Tree-of-Thought** : Yao et al., "Tree of Thoughts" (2023)
- **Self-Consistency** : Wang et al., "Self-Consistency Improves Chain of
  Thought Reasoning in Language Models" (2022)
- **NDCG metric** : Järvelin & Kekäläinen, "Cumulated gain-based evaluation
  of IR techniques" (2002)

---

## 13. Memo for Claude

Quand Claude reprend ce projet, il doit :

1. ✅ Lire ce CLAUDE.md en entier d'abord
2. ✅ Vérifier la phase courante dans la roadmap (actuellement : Phase 7-8)
3. ✅ Respecter les conventions (FR/EN, options multiples, pédagogique)
4. ✅ Toujours livrer code + tests + demo qui passent
5. ✅ Avertir si décision impacte coût > 1$ ou breaking change
6. ✅ Mettre à jour ce fichier si nouvelles décisions architecturales
7. ✅ Suggérer des améliorations pour le mémoire quand pertinent
8. ✅ **Contribution principale = META-LEARNING cross-domaine** — tout le reste sert cet objectif
9. ✅ **Proposer de rédiger des sections du mémoire** à la fin de chaque phase

### Points critiques à ne pas oublier

- Le pipeline complet (Phases 4-5-6) est **fonctionnel**. Ne pas le re-implémenter.
- ProTeGi est dans `src/optimization/` — contribution séparée, **déjà faite**.
- 4 providers LLM dans `src/llm/` — factory transparente pour les agents.
- `WatchState` dans `state.py` = contrat de communication LangGraph.
- Résultats AutoML dans `data/weights/` (**19 fichiers JSON** chargés par QualityCritic).
- Oracle dans `data/oracle/` (**19 domaines**), `promote_oracle_grade2.py --select INDICES` pour sélection manuelle grade-2.
- Meta-learner dans `src/metalearning/meta_learner.py` — **EFFECTIVE_FEATURES = 7 features** (|r|>0.40, n=19).
- Tests : **266 passants** — toujours vérifier avant et après modification.
- ProTeGi experiments dans `data/optimized_prompts/` — 14 runs, 6 configs LLM testées.
- **H3 CONFIRMÉE** : Borda 15/19 (79%), W⁺=149, p=0.0145, r_rb=+0.57 (large). Résultats dans `data/metalearning/phase8c_results.json`.

**Last updated**: Phase 8 complete — H1+H3 confirmed, Wilcoxon done (July 2026)

---

## 14. Phase 3 Results — AutoML Cross-Domain Experiment

### Scoring architecture (Level 3 — 6 features)

```
final_score = w_venue·venue + w_authors·authors + w_impact·impact
            + w_velocity·velocity + w_recency·recency + w_relevance·relevance
```

New Level 3 features (vs Phase 2):
- **velocity_score**: citation momentum (linear, saturation 20 cit/yr). Rewards papers gaining traction.
- **recency_score**: exponential decay with half-life 3 years. Rewards freshness.
- **relevance_score V2**: semantic embeddings (all-MiniLM-L6-v2, 384-dim). Replaces V1 keyword-based.

### Cross-domain NDCG@15 results (H1 confirmed on all 5 domains)

| Domain               | Corpus | Grade-2 | Baseline | Learned | Improvement |
|---|---|---|---|---|---|
| fake_news_detection  | 192    | 17      | 0.0925   | 0.2359  | +155%       |
| federated_learning   | 137    |  5      | 0.1954   | 0.3226  | +65%        |
| graph_neural_networks|  96    |  1      | 0.1598   | 0.1908  | +19%        |
| explainable_ai       | 188    |  5      | 0.0223   | 0.1424  | +538%       |
| medical_img_seg      | 303    |  5      | 0.0000   | 0.0283  | +100%*      |

*Baseline=0 → improvement treated as 100% (bug fix applied)

### Learned weight profiles (key meta-learning signal)

| Domain              | venue | authors | impact | velocity | recency | relevance |
|---|---|---|---|---|---|---|
| fake_news           | 0.027 | 0.061   | 0.037  | **0.698**| 0.104   | 0.024     |
| federated_learning  | 0.201 | **0.295** | 0.175 | 0.138   | 0.153   | 0.038     |
| graph_NN            | 0.204 | 0.128   | 0.084  | 0.221    | **0.329** | 0.034   |
| explainable_ai      | 0.089 | 0.074   | 0.214  | **0.296** | 0.029  | **0.298** |
| medical_img_seg     | 0.079 | 0.169   | **0.278** | **0.271** | 0.173 | 0.031  |

Weight profiles diverge significantly across domains → meta-learning signal confirmed.

### Oracle construction notes

- Oracles built from survey bibliographies via OpenAlex API
- Grade-2 = cited by ≥2 surveys; Grade-1 = cited by 1 survey; Grade-0 = background
- `promote_oracle_grade2.py`: utility script to promote top survey-like grade-1 → grade-2
  when build_oracle.py produces 0 grade-2 (insufficient survey overlap)
- Issue: some DOIs not in OpenAlex (~40% failure rate on academic press DOIs)
- `domains_config.json`: medical `from_year` updated to 2015 (to include U-Net 2015)

### New scripts and files (Phase 3)

- `src/features/relevance_score_v2.py` — SemanticRelevanceScorer (LRU cache, lazy load)
- `src/features/velocity_score.py` — citation momentum score
- `src/features/recency_score.py` — exponential recency decay
- `src/scoring/automl_scorer.py` — extended to 6 features, `_FEATURE_VERSION="f6"`
- `promote_oracle_grade2.py` — CLI utility to fix 0-grade-2 oracles
- `demo_phase3.py` — Phase 3 end-to-end demo (--domain ou --synthetic)
- `demo_relevance_v2.py` — V1 vs V2 relevance comparison
- `data/oracle/*/` — Oracle files for 5 domains
- `data/weights/*/` — Learned weight JSON files per domain
- 266 tests passing (was 56 at Phase 2)

---

## 15. Phases 4-5-6 — Architecture agents (état juin 2026)

### Phase 4 : LLM layer (src/llm/)

4 providers implémentés avec interface unifiée `LLMClient` (Protocol Python) :
- `chat(system, messages, temperature, max_tokens) → LLMResponse`
- `chat_structured(system, messages, schema, ...) → (T, LLMResponse)`

Structured output : Anthropic utilise `tool_use` forcé, OpenAI/Groq/Nebius utilisent `response_format`.
`factory.py` expose `get_llm_for_task(task)` — les agents ne connaissent pas le provider.

### Phase 5 : LangGraph pipeline (src/agents/)

7 agents dans un StateGraph compilé. `WatchState` (TypedDict) = état partagé.
Règle Single Writer / Multiple Readers respectée. `logs` et `errors` utilisent
`Annotated[list, operator.add]` pour l'auto-append entre agents.

| Agent | Pattern | Particularité |
|---|---|---|
| QueryExpander | **ReAct** (4 iter) | Sonde OpenAlex entre chaque LLM call |
| Searcher | — | Multi-query fetch, dedup par ID |
| QualityCritic | — | Charge poids Optuna depuis data/weights/ |
| Summarizer | — | **2 modes** : structured (6 champs) + narrative (prose) |
| Synthesizer | **Reflexion** | Révise si Critic demande |
| Critic | **Reflexion** | Évalue sur 4 axes, force-approve si max iter |
| TrendAnalyst | — | Retry auto si future_perspectives vide |

### Phase 6 : Patterns avancés

**Reflexion** (Shinn et al. 2023) :
- Synthesizer → Critic → Synthesizer (max `MAX_REFLEXION_ITERATIONS=3`)
- Critic évalue : fidelity, completeness, specificity, consistency
- `REFLEXION_MIN_QUALITY="good"` → révision si "poor" ou "acceptable"
- Fallback : si LLM échoue, approuve sans bloquer le pipeline

**ReAct** (Yao et al. 2022) :
- QueryExpander : Pense → Cherche → Observe → Adapte
- Observation = compte d'articles + top concepts + titres exemples
- Toujours démarre par la requête originale exacte (iter 1 forcé)

**ProTeGi** (Pryzant et al. 2023) :
- `src/optimization/protegi_optimizer.py`
- Optimise automatiquement les prompts du Summarizer
- 14 runs dans `data/optimized_prompts/` avec 6 configs LLM :
  haiku-sonnet, gpt4mini-gpt4, llama8b-gpt4, llama8b-sonnet,
  deepseek-v32, nebius-qwen3, en modes structured et narrative

---

## 16. Phase 8 Results — Meta-learning cross-domain (juillet 2026)

### 8a. Corpus d'évaluation — 19 domaines

| Catégorie | Domaines |
|---|---|
| AI classique (16) | fake_news, graph_neural_networks, federated_learning, medical_image_segmentation, neural_architecture_search, explainable_ai, knowledge_graph_embedding, graph_attention_networks, object_detection_deep_learning, sentiment_analysis, transfer_learning, generative_adversarial_networks, anomaly_detection, deep_reinforcement_learning, text_summarization, speech_recognition_asr |
| Hors-AI (3) | computational_biology, medical_nlp, quantitative_finance_ml |

Protocole : Leave-One-Out CV (LOO). Pour chaque domaine test, le meta-learner
est entraîné sur les 18 domaines restants et prédit les poids optimaux pour le domaine test.

### 8b. Meta-features sélectionnées (|r| > 0.40, n=19 après oracle repair)

| Feature | max|r| | Corrélation principale |
|---|---|---|
| `grade2_ratio` | 0.67 | velocity (r=−0.67) |
| `mean_h_index` | 0.61 | relevance (r=−0.61), authors (r=+0.44) |
| `citation_median` | 0.58 | recency (r=+0.58), velocity (r=−0.45) |
| `citation_gini` | 0.56 | venue (r=−0.56) |
| `pct_high_hindex` | 0.52 | relevance (r=−0.52) |
| `pct_recent` | 0.45 | relevance (r=−0.45) |
| `pct_high_cited` | 0.41 | recency (r=+0.41) |

Définies dans `EFFECTIVE_FEATURES` de `src/metalearning/meta_learner.py`.

### 8c. Résultats LOO-CV — H3 (15 méthodes évaluées)

| Méthode | Wins/19 | % | Mean NDCG@15 | vs Default | vs Transfer |
|---|---|---|---|---|---|
| **Borda** ⭐ | **15/19** | **79%** ✓ | **0.282** | +23.1% | +14.2% |
| RRF | 14/19 | 74% | 0.254 | +10.9% | +2.8% |
| En+R (kNN+Ridge) | 13/19 | 68% | 0.253 | +10.4% | +2.4% |
| BayesRidge | 12/19 | 63% | 0.253 | +10.5% | +2.4% |
| En+B (kNN+Bayes) | 12/19 | 63% | 0.258 | +12.5% | +4.5% |
| kNN(1) | 8/19 | 42% | 0.259 | +13.0% | +4.9% |
| GPR | 7/19 | 37% | 0.235 | +2.6% | −4.9% |
| Transfer Direct | (ref) | — | 0.247 | +7.7% | — |
| Default weights | (ref) | — | 0.229 | — | — |
| Optuna* | upper bound | — | 0.387 | +68.7% | — |

*Optuna = poids réels appris sur le domaine test (borne supérieure théorique)

Domaines où Borda **échoue** (4/19) : Transfer Learning, Fake News, Speech ASR, Anomaly Detection.
Ces domaines ont des profils atypiques non capturés par les 7 meta-features actuelles.

### 8d. Test de Wilcoxon signé (H3)

| Méthode | W⁺ | p-value | r_rb | Effet | Significatif (α=0.05) ? |
|---|---|---|---|---|---|
| **Borda** | **149.0** | **0.0145** | **+0.568** | **large** | **✓ OUI** |
| En+R | 130.5 | 0.077 | +0.368 | medium | non |
| RRF | 131.0 | 0.078 | +0.379 | medium | non |
| BayesRidge | 127.5 | 0.095 | +0.345 | medium | non |
| kNN(1) | 96.0 | 0.492 | +0.011 | small | non |
| GPR | 87.5 | 0.619 | −0.143 | small | non |

Script : `phase8d_wilcoxon.py`. Graphe : `data/plots/phase8d_wilcoxon.png`.

### Interprétation pour le mémoire (Section 8.4)

> "Un test de Wilcoxon signé (unilatéral, n=19) confirme que Borda améliore
> significativement le Transfer Direct (W⁺=149, p=0.014, r_rb=+0.57, effet large).
> Aucune autre méthode n'atteint le seuil α=0.05, bien que En+R (p=0.077) et
> RRF (p=0.078) montrent des effets medium prometteurs. La puissance statistique
> limitée (n=19) explique ces résultats marginaux — augmenter à n≥30 domaines
> renforcerait les conclusions."

### Oracle repair — leçon apprise

Les 3 domaines hors-AI avaient des oracles trop faibles au départ (1-3 grade-2).
Solution : `promote_oracle_grade2.py --select "8,11,14,15,23,31,33,34,39" --force --apply`
pour `computational_biology` (1 → 10 grade-2). Cela a stabilisé les corrélations
et révélé `grade2_ratio` comme meta-feature critique (|r|=0.67).
