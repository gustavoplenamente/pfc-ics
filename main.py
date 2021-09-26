from ics import ICS


def main():
    ics = ICS()
    scores = []
    total = 10

    for i in range(total):
        print(f"{i}/{total}")
        score = ics.fit()
        scores.append(score)
        print("{}: {:.2f}".format(i + 1, score * 100))

    print(f"mean: {sum(scores) / len(scores)}")
    print(f"max: {max(scores)}")
    print(f"min: {min(scores)}")

    # print("Predicting...")
    # prediction, prob = ics.predict(333)
    # print(f"Prediction: {prediction}")
    # print("Prob: {:.2f}%".format(prob * 100))


if __name__ == "__main__":
    main()
