import requests
import pandas as pd
from bs4 import BeautifulSoup


def get_top_three_cast(imdb_id):
    url = f'https://www.imdb.com/title/tt{imdb_id}/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta per {imdb_id}: {e}")
        return {'title': "N/A", 'imdbId': imdb_id, 'cast': []}

    if response.status_code != 200:
        print(f"Errore {response.status_code} per {url}")
        return {'title': "N/A", 'imdbId': imdb_id, 'cast': []}

    soup = BeautifulSoup(response.text, 'html.parser')

    # Trova il titolo del film
    title_tag = soup.find('span', {"data-testid": "hero__primary-text"})
    title = title_tag.text.strip() if title_tag else "N/A"

    # Trova i primi tre attori dentro la classe specificata
    cast_section = soup.find('div',
                             class_="ipc-sub-grid ipc-sub-grid--page-span-2 ipc-sub-grid--wraps-at-above-l ipc-shoveler__grid")
    cast = []
    if cast_section:
        cast_links = cast_section.find_all('a', class_="ipc-lockup-overlay ipc-focusable", limit=3)
        cast = [link['aria-label'] for link in cast_links]

    return {'title': title, 'imdbId': imdb_id, 'cast': cast}


if __name__ == '__main__':
    links = pd.read_csv("../CSV_files/links.csv", index_col=False, dtype={"imdbId": str})
    ids = links["imdbId"].values

    data = []
    counter = 0

    for i in ids:
        movie_data = get_top_three_cast(i)
        data.append(
            {"index": counter, "imdbId": i, "title": movie_data['title'], "cast": ", ".join(movie_data['cast'])})
        counter += 1
        print(data[counter - 1])

    df = pd.DataFrame(data)
    #df.to_csv("../CSV_files/cast.csv", index=False, encoding="utf-8", header=[" ", "imdbId", "title", "cast"])

    print("\nFile CSV salvato con successo!")
