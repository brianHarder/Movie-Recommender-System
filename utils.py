from collections import defaultdict
import csv
import numpy as np
import pickle
import tabulate

def load_data():
    item_train = np.genfromtxt('./content_item_train.csv', delimiter=',')
    user_train = np.genfromtxt('./content_user_train.csv', delimiter=',')
    y_train = np.genfromtxt('./content_y_train.csv', delimiter=',')

    item_features = [
    "movie id", "year", "ave rating", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"
    ]
    user_features = [
    "user id", "rating count", "rating ave", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"
    ]

    item_vecs = np.genfromtxt('./content_item_vecs.csv', delimiter=',')

    movie_dict = defaultdict(dict)
    count = 0
    with open('./content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open('./content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)


def print_data(x_train, features, vs, u_s, maxcount=5, user=True):
    if user:
        flist = [".0f", ".0f", ".1f",
                 ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f",
                 ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

    head = features[:vs]
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    header = head + genres
    disp = [split_str(header, 5)]
    count = 0
    for i in range(maxcount):
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)
                    ])
    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
    return table


def split_str(ifeatures, smax):
    result = []
    for s in ifeatures:
        if not ' ' in s:
            if len(s) > smax:
                mid = int(len(s)/2)
                s = s[:mid] + " " + s[mid:]
        result.append(s)
    return result


def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    user_vec_found = False
    for i in range(len(user_train)):
        if user_train[i, 0] == user_id:
            user_vec = user_train[i]
            user_vec_found = True
            break
    if not user_vec_found:
        print(f"did not find uid {user_id} in user_train")
    num_items = len(item_vecs)
    user_vecs = np.tile(user_vec, (num_items, 1))

    y = np.zeros(num_items)
    for i in range(num_items):
        movie_id = item_vecs[i, 0]
        if movie_id in user_to_genre[user_id]['movies']:
            rating = user_to_genre[user_id]['movies'][movie_id]
        else:
            rating = 0
        y[i] = rating

    return(user_vecs, y)


def print_existing_user(y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10):
    count = 0
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]
    count = 0
    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:
            if count == maxcount:
                break
            count += 1
            movie_id = items[i, 0].astype(int)

            offsets = np.nonzero(items[i, ivs:] == 1)[0]
            genre_ratings = user[i, uvs + offsets]
            disp.append([y_p[i, 0], y[i, 0],
                         user[i, 0].astype(int),
                         np.array2string(genre_ratings,
                                         formatter={'float_kind':lambda x: "%.1f" % x},
                                         separator=',', suppress_small=True),
                         items[i, 2].astype(float),
                         movie_id,
                         movie_dict[movie_id]['title'],
                         movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])
    return table


def print_pred_movies(y_p, item, movie_dict, maxcount=10):
    count = 0
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(maxcount):
        movie_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    return tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
