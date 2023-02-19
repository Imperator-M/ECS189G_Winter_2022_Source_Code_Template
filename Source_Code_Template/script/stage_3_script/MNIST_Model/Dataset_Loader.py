import pickle
import matplotlib.pyplot as plt

def loader(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    print("Training Set Size:", len(data["train"]))
    print("Testing Set Size:", len(data["test"]))
    return data
