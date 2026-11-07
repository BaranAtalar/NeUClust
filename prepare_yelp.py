"""Build data/yelp.npz from the Yelp Open Dataset (https://www.yelp.com/dataset).

    python prepare_yelp.py /path/to/yelp_dataset

Only yelp_academic_dataset_business.json and yelp_academic_dataset_review.json are read.

* Base arms are the restaurants (businesses with the "Restaurants" category). Each one is
  described by the TF-IDF of its category names, keeping the terms whose total weight over
  all restaurants is at least 50, and is assigned to one of 20 KMeans clusters.
* Each round a user arrives (users are taken in order of their first restaurant review).
  A user's preference for a term is the sum of the star ratings they gave to restaurants
  whose category equals that term, divided by their number of restaurant reviews.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_CLUSTERS = 20
MIN_TERM_WEIGHT = 50


def read_restaurants(path):
    business_ids, categories = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            business = json.loads(line)
            cats = business.get("categories")
            if cats and "Restaurants" in cats.split(", "):
                business_ids.append(business["business_id"])
                categories.append(cats)
    return business_ids, categories


def restaurant_features(categories):
    corpus = [" ".join(c.split(", ")).replace("&", "") for c in categories]
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(corpus).toarray()
    keep = features.sum(axis=0) >= MIN_TERM_WEIGHT
    return features[:, keep], tfidf.get_feature_names_out()[keep].astype(str)


def user_preferences(review_path, restaurant_terms, num_terms, num_users):
    """Preference vectors of the first `num_users` users that reviewed a restaurant."""
    reviewed, first_stars = {}, {}
    with open(review_path, encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            business, user = review["business_id"], review["user_id"]
            if business not in restaurant_terms:
                continue
            if user not in reviewed:
                if len(reviewed) == num_users:
                    continue
                reviewed[user], first_stars[user] = [], {}
            reviewed[user].append(business)
            first_stars[user].setdefault(business, review["stars"])

    prefs = np.zeros((len(reviewed), num_terms))
    for row, (user, businesses) in enumerate(reviewed.items()):
        for business in businesses:
            for term in restaurant_terms[business]:
                prefs[row, term] += first_stars[user][business]
        prefs[row] /= len(businesses)
    return prefs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yelp_dir", type=Path, help="folder with the Yelp Open Dataset JSON files")
    parser.add_argument("--num-users", type=int, default=1001,
                        help="number of users to keep (round t uses user t + 1)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "yelp.npz")
    args = parser.parse_args()

    print("Reading restaurants...")
    business_ids, categories = read_restaurants(args.yelp_dir / "yelp_academic_dataset_business.json")
    features, terms = restaurant_features(categories)
    print(f"{len(business_ids)} restaurants, {len(terms)} category terms")

    print("Clustering restaurants...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(features)
    order = np.argsort(clusters, kind="stable")

    print("Reading reviews (this takes a few minutes)...")
    term_index = {term: i for i, term in enumerate(terms)}
    restaurant_terms = {
        b: [term_index[c.lower()] for c in cats.split(", ") if c.lower() in term_index]
        for b, cats in zip(business_ids, categories)
    }
    prefs = user_preferences(args.yelp_dir / "yelp_academic_dataset_review.json",
                             restaurant_terms, len(terms), args.num_users)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, restaurant_features=features[order], clusters=clusters[order],
                        business_ids=np.array(business_ids)[order], user_prefs=prefs, terms=terms)
    print(f"Saved {args.out} ({len(prefs)} users)")


if __name__ == "__main__":
    main()
