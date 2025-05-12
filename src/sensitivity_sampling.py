from sklearn.cluster import KMeans


class SensitivitySampling:
    def __init__(self, points, k=10):
        self.points = points
        self.k = k

    def sample(self):
        results = KMeans(n_clusters=self.k).fit(self.points)

        # TODO sensitivity sampling

        return results
