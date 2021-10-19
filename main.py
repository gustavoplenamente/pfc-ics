import time

from ics import ICS


def main():
    ics = ICS(followers_weight=0.0005, followed_by_weight=0.0005)
    scores = []
    total = 1

    init = time.time()
    for i in range(total):
        print(f"{i}/{total}")
        score = ics.fit(test_size=0.3)
        scores.append(score)
        print("{}: {:.2f}".format(i + 1, score * 100))
    end = time.time()

    print(f"mean: {sum(scores) / len(scores)}")
    print(f"max: {max(scores)}")
    print(f"min: {min(scores)}")

    timespan = end - init
    print(f"Timespan for {total} repetitions: {timespan} seconds.")

    # print("Predicting...")
    # prediction, prob = ics.predict(333)
    # print(f"Prediction: {prediction}")
    # print("Prob: {:.2f}%".format(prob * 100))


if __name__ == "__main__":
    main()
