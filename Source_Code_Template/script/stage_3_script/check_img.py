import pickle
import matplotlib.pyplot as plt

if 1:
    f = open("ORL", "rb")
    data = pickle.load(f)
    f.close()
    print("Training Set Size:", len(data["train"]))
    print("Testing Set Size:", len(data["test"]))

    for pair in data["train"]:
        plt.imshow(pair["image"], cmap="Greys")
        plt.show()
        print(pair["label"])