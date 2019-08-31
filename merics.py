from lxml import etree
import argparse
from pathlib import Path
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=Path, help="Path to 'out_data' directory")
parser.add_argument('-g', type=Path, help="Path to 'data' directory")
parser.add_argument('-t', nargs='?', type=int, default=100, choices=range(101),
                    help="Set threshold for square in percents")
parser.add_argument('-v', action="store_const", const="full", help="Print full info about matrix")
parser.add_argument('-s', action="store_const", const="basic", help="Print basic info about matrix")


class Metrics():
    """Class for calculating metrics of Neural Networks"""

    def __init__(self, path_data, path_out_data, threshold=100, output=None):
        self.all_xml_out_data = path_out_data.rglob('*.xml')
        self.all_xml_data = path_data.rglob('*.xml')
        self.threshold = threshold
        self.output = output
        self.dict_out_data = {}
        self.dict_data = {}
        self.full_matrix = defaultdict(lambda: {'tp': [], 'tn': [], 'fp': [], 'fn': []})
        self.matrix = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
        self.TP, self.TN, self.FP, self.FN = (0,) * 4
        self.ACC, self.TPR, self.FPR = (0,) * 3

    def extract(self):
        """Extract values of bndbox from each xml to the dict"""
        for out_data, data in zip(self.all_xml_out_data, self.all_xml_data):
            out_data_xml_tree = etree.parse(str(out_data))
            data_xml_tree = etree.parse(str(data))
            self.dict_out_data[out_data.name] = [{it.tag: int(it.text) for it in list(item)} for item in
                                       out_data_xml_tree.iter('bndbox')]
            self.dict_data[data.name] = [{it.tag: int(it.text) for it in list(item)} for item in
                                       data_xml_tree.iter('bndbox')]
        return self.dict_out_data, self.dict_data

    def full_matrix_square(self):
        """Create dict with values bndbox as TP, TN, FP, FN for each xml data == out_data"""
        for data in self.dict_data:
            for out_data in self.dict_out_data:
                if data == out_data:
                    self.full_matrix[out_data]['fp'] = self.dict_out_data[out_data][:]
                    for val_data in self.dict_data[data]:
                        for val_out_data in self.dict_out_data[out_data]:
                            rect_data = (val_data['xmax'] - val_data['xmin']) * (val_data['ymax'] - val_data['ymin'])
                            dx = min(val_data['xmax'], val_out_data['xmax']) - max(val_data['xmin'], val_out_data['xmin'])
                            dy = min(val_data['ymax'], val_out_data['ymax']) - max(val_data['ymin'], val_out_data['ymin'])
                            if dx >= 0 and dy >= 0:
                                margin = dx * dy
                                if margin / rect_data >= self.threshold / 100:
                                    self.full_matrix[out_data]['tp'].append(([val_data, val_out_data]))
                                    self.full_matrix[out_data]['fp'].remove(val_out_data)
                                    break
                        else:
                            self.full_matrix[out_data]['fn'].append(val_data)
        return self.full_matrix

    def simple_matrix(self):
        """Create a dict with amount of TP, TN, FP, FN for each xml"""
        for k in self.full_matrix:
            for p in self.matrix[k]:
                self.matrix[k][p] = len(self.full_matrix[k][p])
        return self.matrix

    def calc_metrics(self):
        """Calculating a confusion matrix and metrics"""
        terms = Counter()
        for term in self.matrix.values():
            terms.update(term)

        self.TP, self.TN, self.FP, self.FN = terms.values()
        self.ACC = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.TPR = self.TP / (self.TP + self.FN)
        self.FPR = self.FP / (self.FP + self.TN)

    def show_res(self):
        """Print a result"""
        print(f"{'=' * 7} \nTP: {self.TP} \nTN: {self.TN} \nFP: {self.FP} \nFN: {self.FN}")
        print(f"{'=' * 7} \nAC: {self.ACC:.2f} \nTPR: {self.TPR} \nFPR: {self.FPR}\n")

        if 'basic' in self.output:
            print(*((k, v) for k, v in self.matrix.items()), sep='\n')
        elif 'full' in self.output:
            print(*((k, v) for k, v in self.full_matrix.items()), sep='\n')


if __name__ == '__main__':
    args = parser.parse_args()
    a = Metrics(args.d, args.g, threshold=args.t, output=(args.v, args.s))
    a.extract()
    a.full_matrix_square()
    a.simple_matrix()
    a.calc_metrics()
    a.show_res()
