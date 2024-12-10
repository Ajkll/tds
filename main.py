import wikipedia
from requests import get
import json
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial.distance import euclidean
import warnings
from datetime import date, datetime, timedelta
import argparse
import numexpr as ne
import os  # Pour vérifier l'existence du fichier
import nltk  # Pour gérer les stopwords
import platform
import subprocess

# Télécharger les stopwords si nécessaire
try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Création du fichier en_wikipedia_most_common.txt s'il n'existe pas
if not os.path.exists("en_wikipedia_most_common.txt"):
    print("Le fichier 'en_wikipedia_most_common.txt' est manquant. Création automatique...")
    common_words = stopwords.words('english')[:100]  # Limiter aux 100 premiers mots
    with open("en_wikipedia_most_common.txt", "w") as f:
        f.write("\n".join(common_words))
    print("Fichier 'en_wikipedia_most_common.txt' créé avec succès.")

# Analyse des arguments
parser = argparse.ArgumentParser(prog="Projet traitement de signaux", description="Clustering et graphes des pages les plus populaires de Wikipedia")

parser.add_argument("--day", default=str(date.today()-timedelta(days=1))) 
parser.add_argument("--day_start", default=str(date.today()-timedelta(days=31)))
parser.add_argument("--day_end", default=str(date.today()-timedelta(days=1)))
parser.add_argument("--page_number", default=100)
parser.add_argument("--cluster_number", default=10) # 1/10 du nombre de pages marche bien
args = parser.parse_args()
page_number = int(args.page_number)
cluster_number = int(args.cluster_number)
date_format = "%Y-%m-%d"
date = datetime.strptime(args.day, date_format)
graph_start = datetime.strptime(args.day_start, date_format)
graph_end =  datetime.strptime(args.day_end, date_format)

# on filtre les mots les plus courants car ims sont trop génériques
words_reject = open("en_wikipedia_most_common.txt").read().split()[:100] #max 10000

# téléchargement des statistiques des vues
headers = {'User-Agent': 'clustering project'}
def dl_stats(date):
    if date.day < 10:
        day_url = "0"+str(date.day)
    else:
        day_url = date.day
    if date.month < 10:
        month_url = "0"+str(date.month)
    else:
        month_url = date.month
    response = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date.year}/{month_url}/{day_url}", headers=headers)
    return json.loads(response.text)["items"][0]["articles"]

# enlève la ponctuation, les majuscules, les nombres et les charactères spéciaux et on met le résultat dans une liste
def words_process(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char == ' ')
    text = text.split()
    text = list(filter(lambda word: word not in words_reject, text))
    return text

# téléchargement des données d'une page
def scan(title):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            page = wikipedia.page(title, auto_suggest=False)
    except: # les pages de redirection posent problème avec la librairie et génèrent une erreur
        return None
    summary = words_process(page.summary)
    text = words_process(page.content)
    links = page.links
    refs = page.references
    categories = page.categories
    
    return {"title": title, "summary": summary, "text": text, "links": links, "refs": refs, "categories": categories}

# crée un index de tous les mots dans toutes les pages
def create_dict(words_joined):
    words = {}
    w_count = []
    w_num = 0
    for word in words_joined:
        if word not in words:
            words[word] = w_num
            w_count.append(1)
            w_num += 1
        else:
            w_count[words[word]] += 1
    return (words, w_count, w_num)

# crée le vecteur qui représente une page
def create_vector(full_list, w_num, page_list, weight, w_count):
    vect = [0] * w_num
    page_length = len(page_list)
    for word in page_list:
        vect[full_list[word]] += w_count[full_list[word]]
    vect = ne.evaluate("log(v*w/l+1)", local_dict={'v': vect, 'w': weight, 'l': page_length}) # cette librairie accélère le calcul
    return vect

# gère la création des données pour le modèle
def make_data(pages, string, weight):
    all_words = []
    for page in pages:
        all_words += page[string]
    full_list, words_count, words_tot = create_dict(all_words)
    vector = np.array([create_vector(full_list, words_tot, page[string], weight, words_count) for page in pages])
    return vector

# filtre les pages spéciales ect
def filt(pages):
    pages_out = []
    for page in pages:
        if page["article"] == "Main_Page" or page["article"].startswith("Special:")\
           or page["article"].startswith("Wikipedia:") or page["article"].startswith("Category:")\
           or page["article"].startswith("Portal:") or page["article"].startswith("Template:")\
           or page["article"].startswith("Help:"):
            continue
        pages_out.append(page)
        if len(pages_out) == page_number:
            return pages_out
    print("Attention : le nombre de page demandé est trop grand, seulement", len(pages_out), "disponibles")
    return pages_out


filtered = filt(dl_stats(date))
page_number = len(filtered)

try:
    f = open("pages_cache.json", 'r') # fichier de cache pour ne pas re-télécharger tout à chaque fois
    scans = json.load(f)
except:
    scans = []

titles = [i["title"] for i in scans]

# cette étape peut prendre pas mal de temps si les pages ne sont pas dans le cache
c = 0
for i in filtered:
    if i["article"] in titles:
        continue
    scanned = scan(i["article"])
    if scanned is None:
        continue
    scans.append(scanned)
    print("Téléchargement de", scanned["title"])
    
    c += 1
    if c%20 == 0: # dump périodique
        f = open("pages_cache.json", 'w')
        json.dump(scans, f)

f = open("pages_cache.json", 'w')
json.dump(scans, f)

titles = [i["article"] for i in filtered]
scans = [page for page in scans if page["title"] in titles] # ceux dont on a besoin

pages_titles = [i["title"] for i in scans]

# les données finales pour le clustering
vectors = np.concatenate((make_data(scans, "summary", 5),
                          make_data(scans, "text", 1),
                          make_data(scans, "links", 10),
                          make_data(scans, "refs", 2),
                          make_data(scans, "categories", 5)), axis=1)


print("\nConstruction du modèle...\n")
model = KMeans(n_clusters=cluster_number, algorithm = "lloyd")
model.fit(vectors)

# trouver quelles pages sont dans quel cluster
clusters = [{"pages" : []} for i in range(cluster_number)]
index = [[] for i in range(cluster_number)]
for i in range(len(scans)):
    clusters[model.labels_[i]]["pages"].append(scans[i])
    index[model.labels_[i]].append(i)

# trouver le titre de la page la plus au centre de chaque cluster 
centers = []
for i in range(cluster_number):
    center = model.cluster_centers_[i]
    min_dist_index = np.argmin([np.linalg.norm(vectors[j] - center) for j in index[i]])
    min_dist_point = index[i][min_dist_index]
    centers.append(pages_titles[min_dist_point])

# affichage des clusters
titles = []
for i in range(cluster_number):
    print("Cluster " + str(i+1) + " : " + centers[i])
    print("Pages : ", end='')
    titles.append([])
    for page in clusters[i]["pages"]:
        titles[-1].append(page["title"])
        print(page["title"] + " ", end='')
    print('\n')

print("Téléchargement des données pour les graphes...")
stats = []
date_dl = graph_start
c = 0 # le nombre de jours dans la période
while date_dl <= graph_end:
    c += 1
    stats.append(dl_stats(date_dl))
    date_dl += timedelta(days=1)

# remplissage des données pour le graphe
g_data = [[0 for i in range(c)] for j in range(cluster_number)]
for stat_n in range(c):
    for page in stats[stat_n]:
        for cl in range(cluster_number):
            if page["article"] in titles[cl]:
                g_data[cl][stat_n] += page["views"]
                break

# Demander le type de graphique à l'utilisateur
graph_type = input("Quel type de graphique voulez-vous afficher ? (courbe/barres) : ").strip().lower()


# Affichage du graphique en fonction du choix
if graph_type == "barres":
    # Affichage en bâtonnets
    for cl in range(cluster_number):
        plt.bar(range(c), g_data[cl], label=centers[cl], alpha=0.7)
    plt.legend()
    plt.title("Vues de chaque cluster (bâtonnets)")
    plt.xlabel("Jours après " + str(graph_start)[:10])
    plt.ylabel("Vues")
elif graph_type == "courbe":
    # Affichage en courbes
    for cl in range(cluster_number):
        plt.plot(g_data[cl], label=centers[cl])
    plt.legend()
    plt.title("Vues de chaque cluster (courbes)")
    plt.xlabel("Jours après " + str(graph_start)[:10])
    plt.ylabel("Vues")
else:
    print("Choix invalide. Affichage par défaut en courbes.")
    for cl in range(cluster_number):
        plt.plot(g_data[cl], label=centers[cl])
    plt.legend()
    plt.title("Vues de chaque cluster (courbes)")
    plt.xlabel("Jours après " + str(graph_start)[:10])
    plt.ylabel("Vues")

# Affichage ou sauvegarde
try:
    plt.show()
except Exception as e:
    print(f"Impossible d'afficher le graphique (environnement sans interface graphique). Enregistrement dans un fichier...")
    file_name = "graph_clusters.png"
    plt.savefig(file_name)
    print(f"Graphique enregistré sous '{file_name}'.")

    # Ouverture automatique du fichier enregistré
    try:
        if platform.system() == "Windows":
            os.startfile(file_name)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_name])
        else:  # Linux et autres
            subprocess.run(["xdg-open", file_name])
        print(f"Fichier '{file_name}' ouvert automatiquement.")
    except Exception as open_error:
        print(f"Impossible d'ouvrir le fichier automatiquement : {open_error}")
