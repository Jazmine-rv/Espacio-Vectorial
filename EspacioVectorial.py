# Importamos las librerías necesarias
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore (esta bien instalado, solo es un error de vsc)

#pip install seaborn 

# Documentos 
documents = [
    "El veloz zorro marrón salta sobre el perro perezoso.",
    "Un perro marrón persiguió al zorro.",
    "El perro es perezoso."
]

# Convertimos los documentos a vectores usando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculamos la similitud del coseno entre los documentos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Graficamos la matriz de similitud con seaborn
plt.figure(figsize=(8, 6))  
sns.heatmap(
    cosine_sim, 
    annot=True,  
    cmap="pink", 
    xticklabels=[f"Doc {i+1}" for i in range(len(documents))],
    yticklabels=[f"Doc {i+1}" for i in range(len(documents))]
)
plt.title("Matriz de Similitud del Coseno entre Documentos")
plt.show()
