import torch
import torch.nn.functional as func

def tloader(x_train, y_train, x_test, y_test, data):
    # Convert data into tensors
    #x_train = []
    for i in range(len(data["train"])):
        temp = data["train"][i]["image"]
        result = temp[:, :, 0]
        x_train.append(torch.FloatTensor(result))

    x_train = torch.stack(x_train)

    print(type(x_train))
    #print(type(x_train[0]))
    print(x_train.shape)
    
    #y_train = []
    for j in range(len(data["train"])):
        y_train.append(data["train"][j]["label"])

    y_train = torch.Tensor(y_train).type(torch.LongTensor)
    print("Y", type(y_train))
    print("Y", y_train[0])
    print("Y", y_train.shape)

    #x_test = []
    for a in range(len(data["test"])):
        temp = data["test"][a]["image"]
        result = temp[:, :, 0]
        x_test.append(torch.FloatTensor(result))

    x_test = torch.stack(x_test)

    #print(type(x_train))
    #print(type(x_train[0]))
    print(x_test.shape)

    #y_test = []
    for b in range(len(data["test"])):
        y_test.append(data["test"][b]["label"])

    y_test = torch.Tensor(y_test)
    #y_train = func.one_hot(y_train.long(), num_classes=10)
    
    return x_train, y_train, x_test, y_test