from ics import ICS


def main():
    print("Instantiating...")
    ics = ICS()
    print("Fitting...")
    ics.fit()

    print("Predicting...")
    prediction, prob = ics.predict(333)
    print(f"Prediction: {prediction}")
    print("Prob: {:.2f}%".format(prob * 100))


if __name__ == "__main__":
    main()
