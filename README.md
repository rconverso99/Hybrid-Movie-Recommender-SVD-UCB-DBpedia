
# Recommender System per Film basato su Linked Open Data e Multi-Armed Bandits

## Descrizione del Progetto
Questo progetto ha come obiettivo la realizzazione di un **sistema di raccomandazione di film** avanzato, ispirato alle principali piattaforme di streaming commerciali.
Il sistema combina **Collaborative Filtering**, **Linked Open Data** (DBpedia) e tecniche di **Multi-Armed Bandit (MAB)** per bilanciare efficacemente *exploration* ed *exploitation*.

Il progetto è stato sviluppato come parte del corso **Intelligent Web**.

---

## Obiettivi
- Costruire un Recommender System sul dominio dei film
- Utilizzare il dataset **MovieLens Small**
- Arricchire i dati tramite interrogazioni **SPARQL** su **DBpedia**
- Integrare algoritmi di **exploration vs exploitation** (UCB)
- Valutare sperimentalmente le performance dei modelli

---

## Dataset e Fonti
- **MovieLens Small**: rating utenti–film
- **IMDb API**: mapping e recupero attori principali
- **DBpedia**: informazioni semantiche su film, attori e generi

---

## Pre-processing dei Dati
### Riduzione della Sparsità
La matrice utente–item presentava una sparsità ~98%.
Sono stati applicati i seguenti filtri:
- Utenti con almeno **200 rating**
- Film valutati almeno **10 volte**

Sparsità risultante: **~79%**

### Mapping MovieLens ↔ DBpedia
Problema: assenza di un identificativo comune.  
Soluzione:
- Uso del campo `imdb_id` fornito da MovieLens
- Recupero titolo corretto tramite API IMDb
- Interrogazione DBpedia via SPARQL
- Creazione del file `movie_mapping.csv`

---

## Architettura del Sistema
<img width="1263" height="603" alt="image" src="https://github.com/user-attachments/assets/02d9a1ed-4b73-420a-a1de-7868667eedf8" />


Pipeline:
1. Pre-processing dati MovieLens
2. Mapping semantico DBpedia
3. Training modelli di raccomandazione
4. Generazione rating stimati
5. Moduli di raccomandazione tematici

---

## Modelli di Raccomandazione

### SVD – Latent Factor Model
- Implementazione tramite libreria **Surprise**
- Fattorizzazione della matrice utente–item
- Ottimo comportamento con dati sparsi
- Raccomandazioni personalizzate

**Iperparametri ottimizzati:**
- Numero fattori latenti
- Regolarizzazione
- Numero di epoche
- Learning rate

<img width="594" height="385" alt="image" src="https://github.com/user-attachments/assets/5b29c2f6-1361-48d3-a660-1d9c62751bb5" />


### Modelli Confrontati
- **KNNBasic**
- **KNNBaseline**
- **SVD** (scelto)

---

## Valutazione delle Performance
Metriche utilizzate:
- **RMSE**
- **MAE**
- **Coefficient of Variation (CV)**

Risultati principali:
- **SVD** migliore su tutte le metriche
- RMSE: **0.847**
- MAE: **0.649**

<img width="1159" height="438" alt="image" src="https://github.com/user-attachments/assets/127adfa6-ee24-4e5d-8d62-b029aecf1cd2" />


---

## Strategie di Raccomandazione

### Sezione "Per Te"
- Top-10 film con rating stimato più alto
- Strategia orientata all’**exploitation**
- Massimizzazione dell’expected utility

<img width="1079" height="659" alt="image" src="https://github.com/user-attachments/assets/e29312cf-c58f-48e6-b84d-4570eeac85a3" />


### Sezione "Scopri di Più"
- Bilanciamento exploration / exploitation
- Algoritmo **Upper Confidence Bound (UCB)**
- Versione personalizzata basata su rating stimato utente


### Sezione "Perché hai visto..."
- Item-based Collaborative Filtering
- **Cosine Similarity** tra film

### Film del Genere Preferito
- Calcolo genere preferito tramite media pesata
- Raccomandazione film con rating stimato più alto

### Attore Preferito
- Uso IMDb per attori principali
- Media ponderata rating-attore
- Raccomandazione film con attore preferito

---

## Risultati Principali
- Raccomandazioni coerenti con gusti utente
- Migliore scoperta di contenuti con UCB
- Integrazione efficace tra CF e dati semantici
- Sistema modulare e facilmente estendibile

---

## Tecnologie Utilizzate
- Python
- Surprise
- SPARQL
- DBpedia
- IMDb API
- Pandas / NumPy

---

## Valore per un Recruiter
Questo progetto dimostra competenze in:
- Machine Learning applicato
- Recommender Systems
- Data Engineering
- Linked Open Data
- Algoritmi di decisione sequenziale
- Analisi sperimentale e valutazione modelli

---
