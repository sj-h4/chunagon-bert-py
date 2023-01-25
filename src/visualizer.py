import holoviews as hv
from holoviews import dim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np

from models.embedding import Embedding

def compress(embeddings: list[Embedding]):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(np.array([embedding.embedding.detach().numpy().copy() for embedding in embeddings]))
    return reduced_embeddings


def cluster(vector):
    kmeans = KMeans(n_clusters=6, random_state=0).fit(vector)
    return kmeans.predict(vector)


def visualize(vectors: list, labels: list):
    data = {
        'x': [vector[0] for vector in vectors],
        'y': [vector[1] for vector in vectors],
        'label': labels
    }
    hv.extension('bokeh')
    points = hv.Points(data, kdims=['x', 'y'], vdims='label').opts(color=dim('label'), width=800, height=800)
    renderer = hv.renderer('bokeh')
    renderer.save(points, 'server/index')
