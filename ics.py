from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name: str):
    path = f"datasets/{dataset_name}"
    return pd.read_csv(path, sep=';', engine='python')


class ICS:

    def __init__(self, laplace_smoothing=0.01, omega=0.5, news_shared_threshold=5,
                 followers_weight=0.5, followed_by_weight=0.5):

        self.__users = get_dataset("users.csv")
        self.__news = get_dataset("news.csv")
        self.__news_users = get_dataset("posts.csv")
        self.__users_followings = get_dataset("follows.csv")

        self.__smoothing = laplace_smoothing
        self.__omega = omega
        self.__news_shared_threshold = news_shared_threshold
        self.__followers_weight = followers_weight
        self.__followed_by_weight = followed_by_weight

    def fit(self, test_size=0.3):
        """
        Training step: calculate the parameters of each user based on Implicit Crowd Signals
        """
        self.__init_params(test_size)
        unique_users = self.__train_news_users["user_id"].unique()

        for user_id in tqdm(unique_users, desc="Training..."):
            self.__calculate_user_reputation(user_id)

        for user_id in tqdm(unique_users, desc="Adding friends data..."):
            user_prob_alpha, user_prob_um_beta = self.__get_improved_user_reputation(user_id)
            self.__users.loc[self.__users["user_id"] == user_id, "probAlphaN"] = user_prob_alpha
            self.__users.loc[self.__users["user_id"] == user_id, "probUmBetaN"] = user_prob_um_beta

        return self.__assess()

    def predict(self, news_id):
        """
        Classify news based on Implicit Crowd Signals
        """
        users_who_shared_the_news = self.__get_users_who_shared(news_id)

        productAlphaN = 1.0
        productUmAlphaN = 1.0
        productBetaN = 1.0
        productUmBetaN = 1.0

        for user_id in users_who_shared_the_news:
            i = self.__get_user_index(user_id)
            productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
            productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

        # bayesian inference
        reputation_news_not_fake = (self.__omega * productAlphaN * productUmAlphaN) * 100
        reputation_news_fake = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

        # calculate probability grade of the prediction
        total = reputation_news_not_fake + reputation_news_fake

        if reputation_news_not_fake >= reputation_news_fake:
            prob = reputation_news_not_fake / total
            return 0, prob  # news classified as not fake
        else:
            prob = reputation_news_fake / total
            return 1, prob  # news classified as fake

    def __init_params(self, test_size):

        news = self.__news[self.__news['is_fake'].notnull()]
        if not len(news.index):
            raise Exception("No news provided")

        # divide 'self.__news_users' in train and test
        labels = news["is_fake"]
        self.__X_train_news, self.__X_test_news, _, _ = train_test_split(news, labels, test_size=test_size,
                                                                         stratify=labels)

        # store in 'self.__train_news_users' the news posted by each user
        self.__train_news_users = pd.merge(self.__X_train_news, self.__news_users, left_on="news_id",
                                           right_on="news_id")
        self.__test_news_users = pd.merge(self.__X_test_news, self.__news_users, left_on="news_id", right_on="news_id")

        # count fake and not fake news in the train dataset
        self.__qtd_V = self.__news["is_fake"].value_counts()[0]
        self.__qtd_F = self.__news["is_fake"].value_counts()[1]

        # filter users which are also in test dataset
        self.__train_news_users = self.__train_news_users[
            self.__train_news_users["user_id"].isin(self.__test_news_users["user_id"])]

        self.__init_users_params()

    def __init_users_params(self):
        totR = 0
        totF = 0
        alphaN = totR + self.__smoothing
        umAlphaN = ((totF + self.__smoothing) / (self.__qtd_F + self.__smoothing)) * (self.__qtd_V + self.__smoothing)
        betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
        umBetaN = totF + self.__smoothing

        probAlphaN = alphaN / (alphaN + umAlphaN)
        probUmAlphaN = 1 - probAlphaN
        probBetaN = betaN / (betaN + umBetaN)
        probUmBetaN = 1 - probBetaN
        self.__users["probAlphaN"] = probAlphaN
        self.__users["probUmAlphaN"] = probUmAlphaN
        self.__users["probBetaN"] = probBetaN
        self.__users["probUmBetaN"] = probUmBetaN
        self.__users["isAccurate"] = False

    def __assess(self):
        """
        Assessment step: assess news based on each user parameters obtained on training step
        """
        predicted_labels = []
        unique_news_ids = self.__test_news_users["news_id"].unique()

        for news_id in tqdm(unique_news_ids, desc="Assessing trained model..."):
            users_who_shared_the_news = self.__get_users_who_shared(news_id)

            productAlphaN = 1.0
            productUmAlphaN = 1.0
            productBetaN = 1.0
            productUmBetaN = 1.0

            for user_id in users_who_shared_the_news:
                index = self.__get_user_index(user_id)

                user_prob_alpha = self.__users.at[index, "probAlphaN"]
                user_prob_um_beta = self.__users.at[index, "probUmBetaN"]

                productAlphaN = productAlphaN * user_prob_alpha
                productUmBetaN = productUmBetaN * user_prob_um_beta

            # bayesian inference
            reputation_news_not_fake = (self.__omega * productAlphaN * productUmAlphaN) * 100
            reputation_news_fake = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

            if reputation_news_not_fake >= reputation_news_fake:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)

        # print the results of the confusion matrix and accuracy
        gt = self.__X_test_news["is_fake"].tolist()
        print("Confusion matrix:")
        print(confusion_matrix(gt, predicted_labels))

        score = accuracy_score(gt, predicted_labels)
        print("Accuracy: {:.2f}%".format(score * 100))

        return score

    def __calculate_user_reputation(self, user_id):
        # get the labels of news posted by user
        labels_of_news_shared_by_user = list(
            self.__train_news_users["is_fake"].loc[
                self.__train_news_users["user_id"] == user_id
                ]
        )

        # calculate the user opinion matrix
        totR = labels_of_news_shared_by_user.count(0)
        totF = labels_of_news_shared_by_user.count(1)
        alphaN = totR + self.__smoothing
        umAlphaN = ((totF + self.__smoothing) / (self.__qtd_F + self.__smoothing)) * (
                self.__qtd_V + self.__smoothing)
        betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
        umBetaN = totF + self.__smoothing

        # calculate user reputation
        probAlphaN = alphaN / (alphaN + umAlphaN)
        probUmAlphaN = 1 - probAlphaN
        probBetaN = betaN / (betaN + umBetaN)
        probUmBetaN = 1 - probBetaN

        self.__users.loc[self.__users["user_id"] == user_id, "probAlphaN"] = probAlphaN
        self.__users.loc[self.__users["user_id"] == user_id, "probBetaN"] = probBetaN
        self.__users.loc[self.__users["user_id"] == user_id, "probUmAlphaN"] = probUmAlphaN
        self.__users.loc[self.__users["user_id"] == user_id, "probUmBetaN"] = probUmBetaN
        self.__users.loc[self.__users["user_id"] == user_id, "isAccurate"] = \
            len(labels_of_news_shared_by_user) >= self.__news_shared_threshold

    def __get_users_who_shared(self, news_id):
        # TODO: check if list() is really necessary
        is_the_news = self.__news_users["news_id"] == news_id
        return list(
            self.__news_users["user_id"].loc[is_the_news]
        )

    def __get_improved_user_reputation(self, user_id):
        index = self.__get_user_index(user_id)

        user_prob_alpha = self.__users.at[index, "probAlphaN"]
        user_prob_um_beta = self.__users.at[index, "probUmBetaN"]

        followers_alpha_probs, followers_um_beta_probs = self.__get_friends_reputation(user_id, friend_type="follower")
        followed_by_user_alpha_probs, followed_by_user_um_beta_probs = self.__get_friends_reputation(
            user_id, friend_type="followed_by")

        prob_alpha = self.__weight_prob_mean(user_prob_alpha, followers_alpha_probs, followed_by_user_alpha_probs)
        prob_um_beta = self.__weight_prob_mean(user_prob_um_beta, followers_um_beta_probs, followed_by_user_alpha_probs)

        return prob_alpha, prob_um_beta

    def __get_user_index(self, userId):
        # TODO: actually, its just user_id - 1
        is_the_user = self.__users["user_id"] == userId
        return self.__users.loc[is_the_user].index[0]

    def __get_users_followed_by(self, user_id):
        is_the_user = self.__users_followings['user_id'] == user_id
        return self.__users_followings[is_the_user]

    def __get_followed_user(self, followed_id):
        is_the_followed_user = self.__users["user_id"] == followed_id
        return self.__users[is_the_followed_user].iloc[0]

    def __get_users_followers(self, user_id):
        follows_the_user = self.__users_followings['following_id'] == user_id
        return self.__users_followings[follows_the_user]

    def __get_follower(self, follower_id):
        is_the_follower = self.__users["user_id"] == follower_id
        return self.__users[is_the_follower].iloc[0]

    def __get_friends_reputation(self, user_id, friend_type):
        if friend_type == "follower":
            friends = self.__get_users_followers(user_id)
        elif friend_type == "followed_by":
            friends = self.__get_users_followed_by(user_id)
        else:
            raise ValueError(f"Invalid friend_type provided: '{friend_type}'. "
                             "Value should be 'follower' or 'followed_by'.")
        friends_alpha_probs = []
        friends_um_beta_probs = []

        for _, row in friends.iterrows():
            friend_id = row["following_id"]
            friend_user = self.__get_followed_user(friend_id)

            is_accurate = bool(friend_user.loc["isAccurate"])
            if is_accurate:
                friends_alpha_probs.append(friend_user.loc["probAlphaN"])
                friends_um_beta_probs.append(friend_user.loc["probUmBetaN"])

        return friends_alpha_probs, friends_um_beta_probs

    def __weight_prob_mean(self, user_prob, followers_probs, followed_by_probs):
        return (user_prob
                + self.__followers_weight * sum(followers_probs)
                + self.__followed_by_weight * sum(followed_by_probs)) / \
               (1
                + self.__followers_weight * len(followers_probs)
                + self.__followed_by_weight * len(followed_by_probs))
