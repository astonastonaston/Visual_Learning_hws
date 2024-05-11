import urllib.request
import tarfile
import os

dataset_urls = {
    "cifar-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar-100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}


def get_dataset(dataset_name):
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    print("Downloading", dataset_name)
    fileobj = urllib.request.urlopen(dataset_urls[dataset_name])

    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        tar.extractall(path=dataset_path)


if __name__ == "__main__":
    get_dataset("cifar-10")

    print("Done")
