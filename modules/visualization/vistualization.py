import matplotlib.pyplot as plt
import seaborn as ssn
from collections import Counter

class Vistualization:
  def data_visutalization(train_set,test_set):

    train_counter = Counter(train_set.classes)
    test_counter = Counter(test_set.classes)
    print(train_counter.items())
    print(test_counter.items())

    tr_keys = list(train_counter.keys())
    te_keys = list(test_counter.keys())
    vals = [train_counter[k] for k in tr_keys]
    vals_te = [test_counter[j] for j in te_keys]

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    ssn.barplot(ax=axes[0], x=tr_keys, y=vals)
    axes[0].set_title('Training set classes distribution')
    axes[0].set(xlabel='Classes (Car and Bike)', ylabel='Count')
    ssn.barplot(ax=axes[1], x=te_keys, y=vals_te)
    axes[1].set_title('Testing set classes distribution')
    axes[1].set(xlabel='Classes (Car and Bike)', ylabel='Count')
    class_names_t = ["Bike", "Car"]
    n_images = 10
    plt.figure(figsize=(10, 10))
    # DictionaryIterator
    images, labels = train_set.next()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names_t[labels[i].astype("uint8")])
        plt.axis("off")
        i+=1
        if i>=(n_images+1):
            break
    plt.tight_layout()
    plt.show()