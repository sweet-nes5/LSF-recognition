import pickle
import numpy as np
from scipy.spatial import distance

crit_nb = 24


class KmeansData:
    def __init__(self, data, model, distances, cluster_stats):
        self.data = data
        self.model = model
        self.distances = distances
        self.cluster_stats = cluster_stats


# saves a KmeansData object into a .pickle file
def save_object(obj, name):
    try:
        with open(name + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


# loads a KmeansData object from a .pickle file
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


# Computes from a list of hand landmarks a list of criterias that caracterize an image of hand
def criterias(landmarks):
    criteria_values = np.zeros(crit_nb)

    nb1 = 4
    nb2 = 3
    nb_t1 = nb1 * (nb2 + 1)
    for i in range(nb1):
        # distance between thumb and other fingers
        criteria_values[i * (nb2+1)] = distance.euclidean(landmarks[4], landmarks[4*(i+2)])

        for j in range(nb2):
            # comparison of y coordinate between the tip of a finger and the other landmarks of the same finger
            criteria_values[i*(nb2+1) + j+1] = landmarks[4 * (i+2)][1] - landmarks[4 * (i+2) - (j+1)][1]

    nb3 = 3
    nb_t2 = nb_t1 + nb2
    for i in range(nb3):
        # distance between the tip of each consecutive finger (other than thumb)
        criteria_values[nb_t1 + i] = distance.euclidean(landmarks[4 * (i+2)], landmarks[4 * (i+3)])

    nb4 = 5
    for i in range(nb4):
        # distance between the palm and the tip of each finger
        criteria_values[nb_t2 + i] = distance.euclidean(landmarks[0], landmarks[4 * (i+1)])

    '''
    for i in range(1, len(landmarks)):
        criteria_values[2*i - 2] = landmarks[0][0] - landmarks[i][0]  # compare x
        criteria_values[2*i - 1] = landmarks[0][1] - landmarks[i][1]  # compare y
    '''
    # print(criteria_values)
    return criteria_values


# Computes some statistics about the cluster centers and the points corresponding to the hand images
def stats(model, data):
    # for each cluster, creates an array containing the distances between the cluster center and the cluster points
    cluster_nb = len(model.cluster_centers_)
    list_distances = [[0.0] for i in range(cluster_nb)]

    for i in range(0, len(model.labels_)):
        cluster = model.labels_[i]
        list_distances[cluster].append(distance.euclidean(model.cluster_centers_[cluster], data[i]))

    # for each cluster, calculates the average distance and its variance
    cluster_stats = np.zeros((2, cluster_nb))

    for i in range(cluster_nb):
        list_distances[i].pop(0)
        print(len(list_distances[i]))
        cluster_stats[0][i] = np.average(list_distances[i])
        cluster_stats[1][i] = np.std(list_distances[i])

    return list_distances, cluster_stats
