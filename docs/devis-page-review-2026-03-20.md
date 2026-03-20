# Code Review — `devis/page.tsx`

**Date :** 2026-03-20
**Fichier :** `frontend/app/(main)/devis/page.tsx`
**Taille initiale :** 2400 lignes → **2075 lignes** après boucles 1-5
**Nouveau fichier :** `useChoiceHandler.ts` (333 lignes)

---

## Modifications effectuées

### Boucle 1 — `b756df6` → `78389c2`

**Commit :** `fix: devis page — missing Fragment keys, remove debug button, use crypto.randomUUID()`

| Quoi | Avant | Après |
|------|-------|-------|
| Fragment keys manquantes | `<>` sans key dans `groupByEnsemble().map()` | `<Fragment key={...}>` — React réconcilie correctement |
| Bouton debug en prod | `[debug] modal` visible par les utilisateurs | Supprimé |
| IDs non-uniques | 15x `Math.random().toString(36).slice(2)` | 15x `crypto.randomUUID()` — garanti unique |

### Boucle 2 — `78389c2` → `5502d4a`

**Commit :** `refactor: extract Spinner component, replace 8 duplicated SVG spinners`

| Quoi | Avant | Après |
|------|-------|-------|
| SVG spinner copié-collé 8 fois | 3-4 lignes de SVG identiques | `<Spinner className="..." />` — composant réutilisable |

### Boucle 3 — `5502d4a` → `bf81819`

**Commit :** `refactor: extract addPosteAndConfirm() helper, deduplicate 4 identical blocks`

| Quoi | Avant | Après |
|------|-------|-------|
| `addPosteToPanierDirect` + confirm dupliqué 4 fois | ~15 lignes identiques par branche | 1 helper `addPosteAndConfirm()`, gère lock + API + confirm + persist |

### Boucle 4 — `bf81819` → `425514a`

**Commit :** `refactor: extract handleChoiceSelect into useChoiceHandler hook`

| Quoi | Avant | Après |
|------|-------|-------|
| `handleChoiceSelect` = 270 lignes, 8 branches if/else | Inline dans DevisPage | `useChoiceHandler.ts` — 1 fonction par type de choix |
| Pattern "affaire sub-choices" dupliqué 2 fois | Code identique dans `relevance` et `poste` | `showAffaireSubChoices()` + `addPosteOrShowAffaires()` partagés |

**page.tsx : -265 lignes**

### Boucle 5 — `425514a` → `4d678a9`

**Commit :** `fix: stale closure in handleSend + recursion depth guard`

| Quoi | Avant | Après |
|------|-------|-------|
| `handleSend` capture `messages` (stale) | `messages` dans closure + deps array | `messagesRef.current` — toujours frais |
| Récursion sans limite | `handleSend` → choice → `handleSend` → ... | `sendDepthRef` max 5 — abort si dépassé |
| `useChoiceHandler` dépend de `messages` | Callback recréé à chaque changement de messages | Utilise `messagesRef` — callback stable |

---

## Problèmes restants (par priorité)

### Moyenne

| # | Problème | Impact |
|---|----------|--------|
| 5 | `SearchWorkspaceModal` utilise des données mock (L99–112) | Code mort |
| 6 | `event.panier` traité 2 fois au `done` | setState redondant |
| 7 | `fromSilent` sur ChatMessage — jamais lu | Dead code |
| 8 | Side effect `setTimeout(async)` dans `setDevisSettings` updater | Anti-pattern React |

### Basse

| # | Problème | Impact |
|---|----------|--------|
| 9 | `void choice` (L127) | Warning supprimé manuellement |
| 10 | `opt.rows!.length` non-null assertion | TS safety |
| 11 | `setTimeout` sans cleanup dans `handleNumBlur` | setState sur composant démonté |
| 12 | `handleGenerateDevis` déclaré mais jamais utilisé | Dead code (TS warning) |

---

## Résumé des commits

| Boucle | Commit | Delta page.tsx | Risque |
|--------|--------|----------------|--------|
| 0 | `b756df6` | — (commit existant) | — |
| 1 | `78389c2` | -20 lignes | Zéro |
| 2 | `5502d4a` | -11 lignes | Zéro |
| 3 | `bf81819` | -40 lignes | Faible |
| 4 | `425514a` | -265 lignes | Moyen |
| 5 | `4d678a9` | +11 lignes (refs) | Moyen |
| **Total** | | **2400 → 2075 (-325)** | |

---

## Architecture post-refactor

```
devis/
├── page.tsx              (2075 lignes — UI + streaming + state)
└── useChoiceHandler.ts   (333 lignes — dispatch 8 types de choix)
```

Prochaines extractions possibles :
- `useDevisChat.ts` — `handleSend` + SSE streaming logic (~250 lignes)
- Sous-composants UI dans des fichiers séparés (`DevisPanel`, `CandidatesPanel`, etc.)
