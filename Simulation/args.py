import argparse

parser = argparse.ArgumentParser(description='Running arguments')

parser.add_argument('--circuit', type=str, default='', help='specify a circuit name, e.g. CSVA')
parser.add_argument('--model', type=str, default='MultiLayerPerceptron', help='specify a model, e.g. MultiLayerPerceptron')
parser.add_argument('--npoints', type=int, default=1, help='number of points to simulate, -1 if all points')

args = parser.parse_args()