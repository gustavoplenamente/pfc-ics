import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name: str):
    path = f"datasets/{dataset_name}"
    return pd.read_csv(path, sep=';', engine='python')


class ICS:

    def __init__(self, laplace_smoothing=0.01, omega=0.5, news_shared_threshold=5):

        self.__users = get_dataset("users.csv")
        self.__news = get_dataset("2_news.csv")
        self.__news_users = get_dataset("3_post.csv")
        self.__users_followings = get_dataset("userFollowingNewId.csv")

        self.__smoothing = laplace_smoothing
        self.__omega = omega
        self.__news_shared_threshold = news_shared_threshold

    def fit(self, test_size=0.3):
        """
        Training step: calculate the parameters of each user based on Implicit Crowd Signals
        """
        self.__init_params(test_size)

        i = 0
        unique_users = self.__train_news_users["user_id"].unique()
        total = len(unique_users)

        for user_id in unique_users:
            i = i + 1
            print("", end="Training progress: {0:.2f}%\r".format(float((i / total) * 100)), flush=True)
            self.__calculate_user_reputation(user_id)

        self.__assess()

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
            i = self.__users.loc[self.__users["user_id"] == user_id].index[0]
            productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
            productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

        # bayesian inference
        reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
        reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

        # calculate probability grade of the prediction
        total = reputation_news_tn + reputation_news_fn

        if reputation_news_tn >= reputation_news_fn:
            prob = reputation_news_tn / total
            return 0, prob  # news classified as not fake
        else:
            prob = reputation_news_fn / total
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
        self.__train_news_users = pd.merge(self.__X_train_news, self.__news_users, left_on="id_news",
                                           right_on="id_news")
        self.__test_news_users = pd.merge(self.__X_test_news, self.__news_users, left_on="id_news", right_on="id_news")

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
        unique_news_ids = self.__test_news_users["id_news"].unique()

        for news_id in unique_news_ids:
            users_who_shared_the_news = self.__get_users_who_shared(news_id)

            productAlphaN = 1.0
            productUmAlphaN = 1.0
            productBetaN = 1.0
            productUmBetaN = 1.0

            for user_id in users_who_shared_the_news:
                user_prob_alpha, user_prob_um_beta = self.__get_user_probs(user_id)

                productAlphaN = productAlphaN * user_prob_alpha
                productUmBetaN = productUmBetaN * user_prob_um_beta

            # bayesian inference
            reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
            reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

            if reputation_news_tn >= reputation_news_fn:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)

        # print the results of the confusion matrix and accuracy
        gt = self.__X_test_news["is_fake"].tolist()
        print(confusion_matrix(gt, predicted_labels))
        print(accuracy_score(gt, predicted_labels))

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

    def __get_users_who_shared(self, id_news):
        return list(
            self.__news_users["user_id"].loc[
                self.__news_users["id_news"] == id_news
            ]
        )

    def __get_user_probs(self, userId):
        i = self.__users.loc[self.__users["user_id"] == userId].index[0]
        user_prob_alpha = self.__users.at[i, "probAlphaN"]
        user_prob_um_beta = self.__users.at[i, "probUmBetaN"]

        user_id = self.__users.at[i, "user_id"]
        is_user = self.__users_followings['user_id'] == user_id
        followings = self.__users_followings[is_user]

        followings_alpha_probs = []
        followings_um_beta_probs = []

        for index, row in followings.iterrows():
            following_id = row["following_id"]

            is_following = self.__users["user_id"] == following_id
            following_row = self.__users[is_following]

            if following_row.shape[0] > 0:
                is_accurate = bool(following_row.iloc[0].loc["isAccurate"])
                if is_accurate:
                    followings_alpha_probs.append(following_row.iloc[0].loc["probAlphaN"])
                    followings_um_beta_probs.append(following_row.iloc[0].loc["probUmBetaN"])

        prob_alpha = (user_prob_alpha + sum(followings_alpha_probs)) / (1 + len(followings_alpha_probs))
        prob_um_beta = (user_prob_um_beta + sum(followings_um_beta_probs)) / (1 + len(followings_um_beta_probs))

        return prob_alpha, prob_um_beta
