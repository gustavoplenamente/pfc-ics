from ics import ICS


def main():
    print("Instantiating...")
    ics = ICS()
    print("Fitting...")
    ics.fit()

    print("Predicting...")
    prediction, prob = ics.predict(333)
    print(f"Prediction: \t{prediction}")
    print("Prob: \t{prob}")


if __name__ == "__main__":
    main()
