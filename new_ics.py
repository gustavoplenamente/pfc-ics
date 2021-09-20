import pandas as pd
from pandas._typing import ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name: str):
    path = f"datasets/{dataset_name}"
    return pd.read_csv(path, sep=';', engine='python')


class NewICS(BaseEstimator, ClassifierMixin):
    """Fake news classifier implementing Implicit Crowd Signals method.
    """

    def __init__(self, smoothing=0.01, omega=0.5):
        self.smoothing = smoothing
        self.omega = omega

    def _set_users(self):
        # Count amount of fake and not fake news
        self.not_fake_count_ = self.y_.value_counts()[0]
        self.fake_count_ = self.y_.value_counts()[1]

        # filtra apenas os usuários que não estão em ambos os conjuntos de treino e teste.
        # self.__train_news_users = self.__train_news_users[
        #     self.__train_news_users["id_social_media_account"].isin(self.__test_news_users["id_social_media_account"])]

        # inicializa os parâmetros dos usuários
        self.users_ = self.X_["user_id"].drop_duplicates()

        total_not_fake = self.not_fake_count_
        total_fake = self.fake_count_

        prob_alpha_n, prob_beta_n, prob_alpha_n_complement, prob_beta_n_complement = self._get_user_probs(total_fake,
                                                                                                          total_not_fake)

        self.users_["probAlphaN"] = prob_alpha_n
        self.users_["probBetaN"] = prob_beta_n
        self.users_["probUmAlphaN"] = prob_alpha_n_complement
        self.users_["probUmBetaN"] = prob_beta_n_complement

    def __assess(self):
        """
        etapa de avaliação: avalia a notícia com base nos parâmetros de cada usuário obtidos na etapa de treinamento.
        """
        predicted_labels = []
        unique_id_news = self.__test_news_users["id_news"].unique()

        for newsId in unique_id_news:
            # recupera os ids de usuário que compartilharam a notícia representada por 'newsId'.
            usersWhichSharedTheNews = list(
                self.__news_users["id_social_media_account"].loc[self.__news_users["id_news"] == newsId])

            productAlphaN = 1.0
            productUmAlphaN = 1.0
            productBetaN = 1.0
            productUmBetaN = 1.0

            for userId in usersWhichSharedTheNews:
                i = self.__users.loc[self.__users["id_social_media_account"] == userId].index[0]

                productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
                productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

            # inferência bayesiana
            reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
            reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

            if reputation_news_tn >= reputation_news_fn:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)

        # mostra os resultados da matriz de confusão e acurácia.
        gt = self.__X_test_news["ground_truth_label"].tolist()
        print(confusion_matrix(gt, predicted_labels))
        print(accuracy_score(gt, predicted_labels))

    def fit(self):
        """
        Etapa de treinamento: calcula os parâmetros de cada usuário a partir do Implict Crowd Signals.
        """
        self._set_users()

        i = 0
        users_unique = self.__train_news_users["id_social_media_account"].unique()
        total = len(users_unique)

        for userId in users_unique:
            i = i + 1
            print("", end="Progresso do treinamento: {0:.2f}%\r".format(float((i / total) * 100)), flush=True)

            # obtém os labels das notícias compartilhadas por cada usuário.
            newsSharedByUser = list(self.__train_news_users["ground_truth_label"].loc[
                                        self.__train_news_users["id_social_media_account"] == userId])

            # calcula a matriz de opinião para cada usuário.
            totR = newsSharedByUser.count(0)
            totF = newsSharedByUser.count(1)
            alphaN = totR + self.__smoothing
            umAlphaN = ((totF + self.__smoothing) / (self.fake_count_ + self.__smoothing)) * (
                    self.not_fake_count_ + self.__smoothing)
            betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
            umBetaN = totF + self.__smoothing

            # calcula as probabilidades para cada usuário.
            probAlphaN = alphaN / (alphaN + umAlphaN)
            probUmAlphaN = 1 - probAlphaN
            probBetaN = betaN / (betaN + umBetaN)
            probUmBetaN = 1 - probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probAlphaN"] = probAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probBetaN"] = probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmAlphaN"] = probUmAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmBetaN"] = probUmBetaN

        self.__assess()
        return self.__users

    def predict(self, id_news):
        """
        Classifica uma notícia usando o ICS.
        """

        # 17/06/2021
        # usersWhichSharedTheNews = self.__dao.get_users_which_shared_the_news(id_news)
        usersWhichSharedTheNews = list(
            self.__news_users["id_social_media_account"].loc[self.__news_users["id_news"] == id_news])

        productAlphaN = 1.0
        productUmAlphaN = 1.0
        productBetaN = 1.0
        productUmBetaN = 1.0

        # 17/06/2021
        for userId in usersWhichSharedTheNews:
            i = self.__users.loc[self.__users["id_social_media_account"] == userId].index[0]
            productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
            productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

        #        for _, row in usersWhichSharedTheNews.iterrows():
        #            productAlphaN   = productAlphaN  * row["probalphan"]
        #            productUmBetaN  = productUmBetaN * row["probumbetan"]

        # inferência bayesiana
        reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
        reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

        # calculando o grau de probabilidade da predição.
        total = reputation_news_tn + reputation_news_fn
        prob = 0

        if reputation_news_tn >= reputation_news_fn:
            prob = reputation_news_tn / total
            return 0, prob  # notícia classificada como legítima.
        else:
            prob = reputation_news_fn / total
            return 1, prob  # notícia classificada como fake.

    def _fit(self, X: ArrayLike, y: ArrayLike):
        # Check that X and y have correct shape
        self._check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Set users data
        self._set_users()

        i = 0
        total = self.users_.shape

        for userId in users_unique:
            i = i + 1
            print("", end="Progresso do treinamento: {0:.2f}%\r".format(float((i / total) * 100)), flush=True)

            # obtém os labels das notícias compartilhadas por cada usuário.
            newsSharedByUser = list(self.__train_news_users["ground_truth_label"].loc[
                                        self.__train_news_users["id_social_media_account"] == userId])

            # calcula a matriz de opinião para cada usuário.
            totR = newsSharedByUser.count(0)
            totF = newsSharedByUser.count(1)
            alphaN = totR + self.__smoothing
            umAlphaN = ((totF + self.__smoothing) / (self.fake_count_ + self.__smoothing)) * (
                    self.not_fake_count_ + self.__smoothing)
            betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
            umBetaN = totF + self.__smoothing

            # calcula as probabilidades para cada usuário.
            probAlphaN = alphaN / (alphaN + umAlphaN)
            probUmAlphaN = 1 - probAlphaN
            probBetaN = betaN / (betaN + umBetaN)
            probUmBetaN = 1 - probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probAlphaN"] = probAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probBetaN"] = probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmAlphaN"] = probUmAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmBetaN"] = probUmBetaN

        self.__assess()

        return self

    def _predict(self, X: ArrayLike):
        # Check if fit has been called
        self._check_is_fit()

        prediction = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    def _predict_proba(self, X, y):
        pass

    def _get_user_probs(self, total_fake, total_not_fake):
        alpha_n = total_not_fake + self.smoothing
        alpha_n_complement = ((total_fake + self.smoothing) / (self.fake_count_ + self.smoothing)) * (
                self.not_fake_count_ + self.smoothing)
        beta_n = (alpha_n_complement * (total_not_fake + self.smoothing)) / (total_fake + self.smoothing)
        beta_n_complement = total_fake + self.smoothing

        prob_alpha_n = alpha_n / (alpha_n + alpha_n_complement)
        prob_alpha_n_complement = 1 - prob_alpha_n
        prob_beta_n = beta_n / (beta_n + beta_n_complement)
        prob_beta_n_complement = 1 - prob_beta_n

        return prob_alpha_n, prob_beta_n, prob_alpha_n_complement, prob_beta_n_complement

    @staticmethod
    def _check_X_y(X: ArrayLike, y: ArrayLike):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same amount of rows.")

        if X.shape[1] != 2:
            raise ValueError("X must have only 2 features.")

        if y.shape[1] != 1:
            raise ValueError("y must have only 1 dimension.")

    def _check_is_fit(self):
        if not self.X_ or not self.y_:
            raise ValueError("Classifier was not fit. Make sure you call the fit method.")
