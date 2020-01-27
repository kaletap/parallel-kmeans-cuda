import argparse
import subprocess
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from typing import List

def plot_points(points: List[List[float]], colors: List[int]) -> None:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(*zip(*points), c=colors)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File with numbers to run kmeans on", type=str, default="data.txt")
    parser.add_argument("labels_file", help="File with labels corresponding to each point in points file", type=str)
    args = parser.parse_args()
    filename = args.file
    points = []
    with open(filename) as f:
        n = float(f.readline().strip())
        for line in f:
            numbers = line.strip().split()
            points.append([float(num) for num in numbers])
    # C++ excecutable has to be run on a GPU server and write labels to file
    with open(args.labels_file) as labels_file:
        line = labels_file.readline().strip()
        labels = [int(label) for label in line.split()]
    plot_points(points, labels)





