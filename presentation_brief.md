# BRIEF DE PRÉSENTATION — Scientific Watch Agent
> **Instructions pour Claude.ai** : À partir de ce document, génère une présentation complète
> de 15 à 20 slides en français pour un jury de PFE (Master en Informatique / IA).
> Format souhaité : slide par slide, avec **titre**, **contenu principal (bullets)** et
> **note orateur** (2-3 phrases pour expliquer chaque slide à voix haute).
> Style : académique, professionnel, sobre. Utilise des émojis avec parcimonie (max 1 par slide).
> La présentation doit durer environ 20 minutes.
> Inclure : une slide de titre, un sommaire, les sections ci-dessous, et une slide de conclusion/questions.

---

## INFORMATIONS GÉNÉRALES

| Champ | Valeur |
|---|---|
| **Titre du projet** | Scientific Watch Agent — Système Multi-Agents de Veille Scientifique |
| **Type** | Projet de Fin d'Études (PFE) — Master Informatique / IA |
| **Durée** | 4 mois |
| **Technologies** | Python 3.10, LangGraph, Anthropic Claude, OpenAI GPT, Optuna, Pydantic v2 |
| **Date** | Mai 2026 |

---

## SECTION 1 — PROBLÉMATIQUE ET MOTIVATION

### Contexte
Les chercheurs académiques en début de thèse font face à une **explosion du volume de publications scientifiques** :
- Plus de 2 millions d'articles publiés par an (toutes disciplines)
- Comprendre un domaine inconnu prend plusieurs semaines de lecture manuelle
- Identifier les tendances émergentes, les gaps de recherche et les axes prometteurs est un processus long et subjectif

### Utilisateur cible
**Chercheurs en début de thèse** qui doivent rapidement :
- Cartographier un domaine inconnu (état de l'art)
- Identifier les approches dominantes et les datasets de référence
- Repérer les gaps de recherche et les directions émergentes
- Filtrer les articles par qualité (éviter le bruit)

### Question de recherche centrale
> Comment concevoir un système multi-agents capable de produire automatiquement une veille scientifique complète et généralisable à n'importe quel domaine de recherche ?

---

## SECTION 2 — VISION ET OBJECTIFS

### Vision en une phrase
Un système multi-agents qui, à partir d'un simple topic de recherche, produit automatiquement une **veille scientifique complète** : articles filtrés par qualité, résumés structurés, synthèse globale, tendances et perspectives.

### Pipeline de bout en bout
```
Topic de recherche
      ↓
  [Recherche]  ──→  OpenAlex API (100K+ articles sources)
      ↓
[Filtrage qualité]  ──→  Score composite : venue + auteurs + impact + pertinence
      ↓
  [Résumés LLM]  ──→  Summarizer Agent (Claude Haiku)
      ↓
[Synthèse globale]  ──→  Synthesizer Agent (Claude Sonnet, 200K contexte)
      ↓
[Analyse de tendances]  ──→  Trend Analyst Agent (Claude Sonnet)
      ↓
  [Rapport PDF]  ──→  Livrable final structuré
```

### Trois critères de succès
1. **Engineering** : code propre, 100+ tests, architecture maintenable
2. **Recherche** : métriques quantitatives, expérimentations rigoureuses
3. **Produit** : livrable utilisable, démo convaincante sur domaines réels

---

## SECTION 3 — ARCHITECTURE COMPLÈTE

### Vue d'ensemble — 9 couches

| Couche | Composant | Rôle |
|---|---|---|
| 1. Sources | `OpenAlexClient` | Récupère les articles via API REST (filtres: date, citations, domaine) |
| 2. Features | `venue_score`, `authors_score`, `impact_score`, `relevance_score` | 4 scores de qualité indépendants |
| 3. Scoring | `QualityScorer` + `AutoMLScorer` (Optuna) | Agrège les features, optimise les poids |
| 4. LLM Abstraction | `ClaudeClient`, `OpenAIClient`, `EnsembleClient` | Interface unifiée multi-modèle |
| 5. Agents | `QueryExpander`, `Summarizer`, `Synthesizer`, `TrendAnalyst`, `Critic` | Intelligence spécialisée |
| 6. Orchestration | LangGraph (state machine) | Coordination, état partagé, routing conditionnel |
| 7. Optimisation | `ProTeGiOptimizer` | Améliore les prompts automatiquement |
| 8. Données | `schemas.py` (Pydantic v2) | Contrat de communication formel |
| 9. Tests | pytest (100+ tests) | Couverture complète, mocks réseau |

### Shared State (LangGraph)
Le cœur de la coordination : un `TypedDict` partagé entre tous les agents, avec la règle **Single Writer / Multiple Readers** — chaque section n'est écrite que par un seul agent, mais peut être lue par tous.

---

## SECTION 4 — COMPOSANTS CLÉS

### 4.1 — Scoring de Qualité (Phase 2)

**4 features indépendantes** calculées sans LLM (rapide, déterministe) :

| Feature | Méthode | Signal |
|---|---|---|
| `venue_score` | Scimago SJR quartile (CSV local 30K venues) | Réputation de la revue |
| `authors_score` | h-index moyen des auteurs (OpenAlex) | Expertise des auteurs |
| `impact_score` | Citations normalisées par âge et domaine | Impact de l'article |
| `relevance_score` | Overlap de mots-clés (V1) / embeddings (V2) | Pertinence topique |

**Score composite** : moyenne pondérée des 4 features
→ Seuil de filtrage configurable (ex: garder top 15 articles sur 50 récupérés)

### 4.2 — Couche LLM Multi-Modèles (Phase 4)

**Abstraction unifiée** via protocole `LLMClient` :
- `chat_structured(system, messages, ResponseModel)` → retourne un objet Pydantic validé
- Supporte : Claude (Anthropic API) + GPT (OpenAI API)
- Prompt caching activé : **90% de réduction sur les inputs récurrents**

**Mapping tâche → modèle (TASK_MODELS)** :

| Tâche | Modèle | Raison |
|---|---|---|
| Query expansion | GPT-4o-mini | Rapide, économique |
| Summarize | Claude Haiku 4.5 | Tâche répétitive, structurée |
| Synthesize | Claude Sonnet 4.6 | Contexte 200K tokens |
| Trend analysis | Claude Sonnet 4.6 | Raisonnement complexe |
| Critic (Reflexion) | Claude Sonnet 4.6 | Analyse nuancée |
| Judge | Claude Opus 4.7 | Meilleur modèle pour évaluer |

### 4.3 — Agents Spécialisés (Phase 5)

**5 agents**, chacun avec une responsabilité unique :

1. **QueryExpander** : transforme un topic en 3-5 requêtes de recherche diversifiées (synonymes, sous-domaines, termes techniques)
2. **Summarizer** : produit un résumé structuré en 6 champs (problème, méthode, dataset, résultats, limitations, contributions) via structured outputs Pydantic
3. **Synthesizer** : synthèse globale sur les N articles (approches dominantes, consensus, divergences) avec contexte 200K tokens
4. **TrendAnalyst** : identifie tendances mature/émergente, gaps de recherche, perspectives futures avec articles de preuve
5. **Critic** : juge la qualité d'une synthèse et fournit des suggestions d'amélioration (pattern Reflexion)

### 4.4 — Pattern Reflexion (Phase 6)

**Inspiré de** : Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)

**Fonctionnement** :
```
Synthesizer génère une synthèse
        ↓
Critic évalue : [score 0-1] + [feedback structuré]
        ↓
  Score < seuil ? ──→ Synthesizer révise avec le feedback
        ↓                        ↑
  Score ≥ seuil ?                └─── (max 3 itérations)
        ↓
  Synthèse finale validée
```

**Valeur** : détecte les hallucinations, améliore la couverture, force la précision factuelle

---

## SECTION 5 — INNOVATIONS TECHNIQUES

### 5.1 — AutoML pour le Scoring (Phase 3 — Optuna)

**Problème** : les poids `[w_venue, w_authors, w_impact, w_relevance]` sont fixés manuellement → suboptimaux selon le domaine.

**Solution** : Optuna (algorithme TPE) apprend les poids optimaux sur un **dataset oracle** labellisé par domaine.

**Processus** :
- Oracle : 50-100 articles labellisés « pertinent » / « non pertinent » par des experts
- Optuna explore l'espace des poids pour maximiser le NDCG@15
- Les poids optimaux sont stockés par domaine

**Résultat attendu (H1)** : les poids Optuna battent les poids manuels avec ≥5% d'amélioration NDCG@15.

### 5.2 — ProTeGi : Optimisation Automatique des Prompts

**Inspiré de** : "Automatic Prompt Optimization with Gradient Descent and Beam Search" (Pryzant et al., 2023)

**Concept** : comme la descente de gradient en deep learning, mais pour les prompts en langage naturel.

**Algorithme** :
```
Pour chaque itération :
  1. Évaluer le prompt courant sur les exemples d'entraînement
  2. Identifier les K pires exemples (gradient signal)
  3. LLM génère un "gradient textuel" (diagnostic des échecs)
  4. LLM propose N prompts candidats améliorés (beam search)
  5. Valider chaque candidat sur le set de validation
  6. Sélectionner le meilleur candidat
  └── Répéter
```

**Métriques d'évaluation** :
- ROUGE-1/2/L (overlap lexical avec la référence)
- LLM Judge : faithfulness [0,1] + coverage [0,1]
- Score composite : `0.4×ROUGE-L + 0.4×faithfulness + 0.2×coverage`

**Résultat réel obtenu** sur PubMed (119K articles) :
- Dataset : 20 exemples entraînement, 10 validation, 3 itérations
- Amélioration composite : +1.2% (légère mais statistiquement cohérente)
- Exemple de changement ProTeGi : ajout de *"including specific figures or statistical findings when available"* pour guider la précision des résultats

**Analyse scientifique du résultat** : l'amélioration modeste est attendue — le prompt initial était déjà bien conçu (rendements décroissants), et ROUGE pénalise les synonymes (limitation connue). Ce résultat est lui-même un **finding intéressant** pour le mémoire.

### 5.3 — Rapport PDF Structuré

**Généré avec ReportLab**, 7 sections dans l'ordre de priorité utilisateur :

| # | Section | Contenu |
|---|---|---|
| 1 | Tendances & Gaps | Tendances (mature/émergente) avec articles de preuve + digest 2-3 phrases |
| 2 | Synthèse globale | Vue d'ensemble, approches dominantes, consensus |
| 3 | Top Articles | Tableau scoré avec venue, auteurs, citations |
| 4 | Résumés structurés | 6 champs par article (problème, méthode, dataset...) |
| 5 | Query Expansion | Requêtes générées avec rationale |
| 6 | Boucle Reflexion | Historique des itérations Critic |
| 7 | Logs d'exécution | Tokens, coûts, durées par agent |

---

## SECTION 6 — CONTRIBUTION SCIENTIFIQUE PRINCIPALE

### Généralisation Cross-Domaine (Méta-Learning) — Phase 8

**La question centrale du mémoire** :
> Un système de veille scientifique peut-il **apprendre à s'adapter automatiquement** à un nouveau domaine de recherche, sans re-entraînement spécifique ?

**Idée centrale** : au lieu d'apprendre des paramètres optimaux *pour un domaine* (ex: détection de fake news), on apprend une **fonction qui prédit les bons paramètres à partir de meta-features du domaine**.

```
meta_features(domaine) ──→ poids_optimaux
```

Exemples de meta-features :
- Densité du vocabulaire technique
- Maturité du champ (moyenne des années de publication)
- Distribution des citations (longue traîne ou concentrée ?)
- Taux d'articles expérimentaux vs théoriques

### Trois niveaux de généralisation

| Niveau | Méthode | Mesure |
|---|---|---|
| 1. Transfer direct | Poids du Domaine A → Domaine B tels quels | Performance relative |
| 2. Fine-tuning rapide | Ajustement sur B avec K exemples (few-shot) | Nb d'exemples nécessaires |
| 3. Méta-apprentissage | `meta_features(B) → poids_B` via modèle entraîné sur A, C, D... | Corrélation prédiction/optimal |

### Cinq hypothèses de recherche

| Hypothèse | Prédiction | Métrique |
|---|---|---|
| **H1** | Poids Optuna > poids manuels sur même domaine | NDCG@15 ≥ +5% |
| **H2** | Transfer direct dégrade de façon prévisible (corrélé à la distance entre domaines) | Dégradation entre 15% et 40% |
| **H3** | Méta-learning > transfer direct | Battu dans ≥75% des cas |
| **H4** | L'architecture multi-agents identifie les goulots de généralisation | Variance inter-agents mesurable |
| **H5** | Claude et GPT ont des profils de généralisation différents | Écart statistiquement significatif |

---

## SECTION 7 — PHASES DU PROJET

### Roadmap complète

| Phase | Statut | Description | Innovation clé |
|---|---|---|---|
| 0 | ✅ Terminé | Setup, structure projet, schemas Pydantic v2 | Contrat de communication formel |
| 1 | ✅ Terminé | Source OpenAlex (API wrapper + pagination) | 100K+ articles accessibles |
| 2 | ✅ Terminé | 4 features de qualité (venue, auteurs, impact, pertinence) | Score composite multi-critères |
| 3 | ✅ Terminé | AutoML (Optuna) pour les poids de scoring | Poids appris vs poids manuels |
| 4 | ✅ Terminé | Couche LLM unifiée (Claude + GPT) | Abstraction multi-provider |
| 5 | ✅ Terminé | Agents + LangGraph orchestration | Pipeline multi-agents complet |
| 6 | ✅ Terminé | Pattern Reflexion + Critic + ProTeGi | Auto-amélioration des prompts |
| 7 | 🔜 En cours | V3 concurrent (Best-of-N, Claude vs GPT, Judge) | Ensemble multi-modèle |
| 8 | 🔜 Planifié | Campagne évaluation multi-domaine + Méta-learning | **Contribution centrale** |

### Métriques de qualité engineering

- **Tests** : 100+ tests pytest, mocks pour tous les appels réseau
- **Architecture** : 9 modules indépendants, responsabilité unique
- **Coût optimisé** : ~$0.21 par run complet (15 articles) grâce au prompt caching
- **Format** : Pydantic v2 valide toutes les données inter-agents

---

## SECTION 8 — STACK TECHNIQUE

### Technologies utilisées

| Catégorie | Outil | Utilisation |
|---|---|---|
| **Langage** | Python 3.10+ | Type hints, modern syntax |
| **Validation** | Pydantic v2 | Schémas de données, structured outputs |
| **Orchestration** | LangGraph | State machine multi-agents |
| **LLM** | Anthropic Claude API | Haiku (résumés), Sonnet (synthèse, Reflexion), Opus (Judge) |
| **LLM** | OpenAI GPT-4o-mini | Query expansion |
| **AutoML** | Optuna (TPE sampler) | Optimisation des poids de scoring |
| **Sources** | OpenAlex API | 250M+ articles, gratuit, REST |
| **Génération PDF** | ReportLab | Rapport structuré 7 sections |
| **Métriques** | ROUGE + LLM Judge | Évaluation des résumés |
| **Tests** | pytest + unittest.mock | 100+ tests, 0 appels réseau réels |

### Principes d'architecture

1. **Single Responsibility** : un module = une responsabilité
2. **Communication via contrat** : Pydantic v2 à chaque frontière d'agent
3. **Configuration centralisée** : `TASK_MODELS` dans `config.py` pilote tout
4. **Testabilité** : fonctions pures, mocks injectables, pas de hardcoding
5. **Coût-conscience** : prompt caching, batch API, tracking tokens/coût par agent

---

## SECTION 9 — RÉSULTATS ET DÉMO

### Exemple de run réel

**Topic** : "fake news detection"
**Configuration** : 30 articles récupérés → 15 retenus après filtrage → 15 résumés → 1 synthèse → analyse tendances → rapport PDF

**Output rapport (extrait Tendances)** :
- **Tendances matures** : Modèles BERT fine-tunés pour la détection de désinformation, détection basée sur les graphes de propagation
- **Tendances émergentes** : LLMs pour la génération + détection simultanée, approches multimodales (texte + image)
- **Gaps identifiés** : Peu d'approches cross-linguistiques, manque de datasets multimodaux annotés
- **Perspectives** : Systèmes de vérification en temps réel, adaptation au contexte culturel

### Exemple de résumé structuré généré

```
Article : "Detecting Fake News with BERT and Graph Networks" (2024)

Problème   : La détection automatique de fake news reste difficile car
             elle nécessite la compréhension du contexte et des relations
             entre sources.
Méthode    : Modèle hybride BERT + Graph Attention Network exploitant
             la structure de propagation des tweets.
Dataset    : FakeNewsNet (22K articles annotés, Twitter propagation).
Résultats  : F1-score de 0.89 sur PolitiFact, +4.2% vs BERT seul.
Limitations: Évalué uniquement sur l'anglais ; performances non testées
             sur des domaines hors politique.
Contributions :
  • Première intégration de BERT avec graphes de propagation pour la
    détection de fake news
  • Dataset augmenté de 15K exemples publiés en open access
```

### Coût d'un run complet
| Configuration | Coût estimé |
|---|---|
| 15 articles, sans Judge | ~$0.04 |
| 15 articles, avec LLM Judge | ~$0.21 |
| ProTeGi 20/10/3 iter | ~$0.47 |
| 50 expériences benchmark | ~$21 |

---

## SECTION 10 — PERSPECTIVES ET TRAVAUX FUTURS

### Phase 7 (prochaine) — V3 Concurrent

**Best-of-N avec LLM Judge** :
- Claude et GPT génèrent des synthèses **en parallèle**
- `EnsembleClient` agrège les réponses
- `JudgeAgent` (Claude Opus) sélectionne ou fusionne le meilleur résultat
- Objectif : répondre à H5 (profils de généralisation Claude vs GPT)

### Phase 8 (finale) — Méta-Learning Cross-Domaine

**Plan d'expérimentation** :
1. Entraîner le système sur 3 domaines sources : NLP, Computer Vision, Bioinformatique
2. Extraire les meta-features de chaque domaine
3. Apprendre la fonction `meta_features → poids_optimaux` (régression/forêt aléatoire)
4. Tester sur 2 domaines cibles : Cybersécurité, Économie computationnelle
5. Comparer : transfer direct vs fine-tuning rapide vs méta-learning

**Domaines de validation prévus** :
- Fake news detection (domaine de test principal)
- Medical imaging segmentation
- Quantum computing optimization
- Climate change modeling
- NLP for low-resource languages

### Impact attendu

Ce travail ouvre la voie vers des **systèmes de veille auto-adaptatifs** :
- Un chercheur peut utiliser le même outil pour n'importe quel domaine
- Le système s'améliore automatiquement avec chaque nouveau domaine exploré
- Transférable à d'autres types d'agents (recommandation, monitoring, etc.)

---

## ANNEXES (informations complémentaires pour les slides)

### Comparaison architecturale avec l'existant

| Système | Approche | Limitation vs ce projet |
|---|---|---|
| Google Scholar Alerts | Basé sur mots-clés, pas de résumé | Pas de filtrage qualité, pas de synthèse |
| Semantic Scholar | Résumés automatiques, pas d'agents | Pas de synthèse multi-articles, pas de tendances |
| Elicit.org | LLM + recherche | Pas de scoring qualité, pas de méta-learning |
| **Notre système** | Multi-agents + AutoML + Méta-learning | — |

### Références bibliographiques clés

1. Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS.
2. Du et al. (2023). *Improving Factuality and Reasoning through Multiagent Debate*. MIT.
3. Pryzant et al. (2023). *Automatic Prompt Optimization with Gradient Descent*. EMNLP.
4. Finn et al. (2017). *Model-Agnostic Meta-Learning (MAML)*. ICML.
5. Järvelin & Kekäläinen (2002). *Cumulated gain-based evaluation of IR techniques*. ACM TOIS.
6. Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.

### Structure du code (résumé)

```
scientific-watch-agent/
├── src/
│   ├── config.py              # TASK_MODELS, poids, seuils
│   ├── schemas.py             # 8 schémas Pydantic v2
│   ├── sources/openalex.py    # API wrapper (pagination, filtres)
│   ├── features/              # 4 modules de scoring
│   ├── scoring/               # Agrégateur + Optuna AutoML
│   ├── llm/                   # ClaudeClient, OpenAIClient, Ensemble
│   ├── agents/                # 5 agents spécialisés
│   └── optimization/          # ProTeGi + métriques ROUGE/Judge
├── tests/                     # 100+ tests pytest
├── export_to_pdf.py           # Générateur de rapport PDF
├── demo_protegi.py            # Démo optimisation de prompts
└── demo_phase*.py             # Démos par phase
```
