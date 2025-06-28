import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.manifold import TSNE

# Carica i dati
ratings = pd.read_csv('../CSV_files/ratings.csv')  # Contiene userId, movieId, rating
movies = pd.read_csv('../CSV_files/movies.csv')  # Contiene movieId, title, genres

# Definiamo il formato dei dati per Surprise
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Creiamo il set di addestramento e di test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Creiamo e addestriamo il modello SVD
svd = SVD(n_factors=80,
                   reg_all=0.06,
                   n_epochs=30,
                   lr_all=0.01)
svd.fit(trainset)

# Seleziona un utente
user_id = int(input("Inserisci uno userId: "))

# Seleziona un film
movie_id_input = int(input("Inserisci un movieId: "))

# Predici le valutazioni per tutti i film per l'utente
all_movie_ids = movies['movieId'].values
predictions = [svd.predict(user_id, movie_id) for movie_id in all_movie_ids]

# Ordina i film in base alle valutazioni predette
predictions.sort(key=lambda x: x.est, reverse=True)

# Seleziona i 10 film con le valutazioni più alte (più affini)
top_10_predictions = predictions[:10]
top_10_movie_ids = [pred.iid for pred in top_10_predictions]

# Seleziona i 30 film con le valutazioni più basse (meno affini)
bottom_30_predictions = predictions[-30:]
bottom_30_movie_ids = [pred.iid for pred in bottom_30_predictions]

# Stampa i film più affini con i loro estimated ratings
print(f"\nTop 10 film raccomandati per l'utente {user_id}:")
print("-" * 50)
for pred in top_10_predictions:
    movie_id = pred.iid
    est_rating = pred.est
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    genres = movies[movies['movieId'] == movie_id]['genres'].values[0]
    print(f"Movie: {title} | Genres: {genres} | Estimated Rating: {est_rating:.2f}")

# Combina i film selezionati, includendo il movieId inserito
selected_movie_ids = top_10_movie_ids + bottom_30_movie_ids
if movie_id_input not in selected_movie_ids:  # Evita duplicati
    selected_movie_ids.append(movie_id_input)

# Filtra solo i movie_id presenti nel trainset e converti in indici interni
valid_movie_ids = []
inner_ids = []
for movie_id in selected_movie_ids:
    try:
        inner_id = trainset.to_inner_iid(movie_id)  # Converte raw movieId in inner ID
        valid_movie_ids.append(movie_id)
        inner_ids.append(inner_id)
    except ValueError:
        # Se il movie_id non è nel trainset, lo saltiamo
        continue

# Ottieni il vettore latente dell'utente
try:
    user_inner_id = trainset.to_inner_uid(user_id)  # Converte raw userId in inner ID
    user_vector = svd.pu[user_inner_id]  # Vettore latente dell'utente
except ValueError:
    print(f"L'utente {user_id} non è presente nel trainset!")
    exit()

# Estrai i fattori latenti per i film selezionati e aggiungi quello dell'utente
movie_vectors = np.array([svd.qi[inner_id] for inner_id in inner_ids])
all_vectors = np.vstack([movie_vectors, user_vector])  # Combina film e utente

# Riduci la dimensionalità con t-SNE
perplexity_value = min(len(all_vectors) - 1, 30)  # Assicurati che perplexity sia <= numero di campioni - 1
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
all_2d = tsne.fit_transform(all_vectors)

# Separa i dati: film e utente
movies_2d = all_2d[:-1]  # Tutti i film
user_2d = all_2d[-1]     # L'ultimo è l'utente

# Creazione del grafico
plt.figure(figsize=(14, 10))

# Crea una griglia per la mappa di calore
x = np.linspace(min(all_2d[:, 0]), max(all_2d[:, 0]), 100)
y = np.linspace(min(all_2d[:, 1]), max(all_2d[:, 1]), 100)
X, Y = np.meshgrid(x, y)

# Calcola la distanza euclidea dalla posizione dell'utente per ogni punto della griglia
Z = np.exp(-((X - user_2d[0])**2 + (Y - user_2d[1])**2) / 10.0)  # Funzione gaussiana

# Plotta la mappa di calore
plt.contourf(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.5)  # Rosso vicino, blu lontano

# Separa i dati per top e bottom (solo per i film validi)
n_top_valid = min(len(top_10_movie_ids), len(valid_movie_ids))  # Numero di top film validi
top_10_2d = movies_2d[:n_top_valid]
bottom_30_2d = movies_2d[n_top_valid:len(top_10_movie_ids) + len(bottom_30_movie_ids)]

# Trova l'indice del movieId inserito tra i valid_movie_ids
movie_input_idx = None
if movie_id_input in valid_movie_ids:
    movie_input_idx = valid_movie_ids.index(movie_id_input)
    movie_input_2d = movies_2d[movie_input_idx]

# Plotta i film più affini (blu scuro)
plt.scatter(top_10_2d[:, 0], top_10_2d[:, 1], c='darkblue', s=100, edgecolors='k',
           label='Film più affini')

# Plotta i film meno affini (rosso chiaro) senza etichette
plt.scatter(bottom_30_2d[:, 0], bottom_30_2d[:, 1], c='lightcoral', s=100, edgecolors='k',
           label='Film meno affini')

# Plotta il vettore dell'utente (verde, stella)
plt.scatter(user_2d[0], user_2d[1], c='green', s=200, marker='*', edgecolors='k',
           label=f'Utente {user_id}')

# Plotta il film selezionato (viola, triangolo) se presente nel trainset
if movie_input_idx is not None:
    title = movies[movies['movieId'] == movie_id_input]['title'].values[0]
    plt.scatter(movie_input_2d[0], movie_input_2d[1], c='purple', s=150, marker='^', edgecolors='k',
               label=f'Film selezionato: {title}')

# Annotazioni solo per i film più affini
for i, movie_id in enumerate(valid_movie_ids[:n_top_valid]):  # Solo i top validi
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    genres = movies[movies['movieId'] == movie_id]['genres'].values[0]
    plt.annotate(f"{title}\n{genres}",
                (movies_2d[i, 0], movies_2d[i, 1]),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points')

# Aggiungi il titolo e gli assi
plt.title(f'Mappa di calore dei gusti dell’utente {user_id}', fontsize=14)
plt.xlabel('t-SNE Dimensione 1', fontsize=12)
plt.ylabel('t-SNE Dimensione 2', fontsize=12)

# Aggiungi la griglia e la legenda
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()