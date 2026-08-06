import editdistance


class CER:
    def __init__(self, remove_space=False):
        self.remove_space = remove_space
        self.reset()

    def reset(self):
        self.edits = 0
        self.total = 0

    def update(self, predictions, references):
        for hypothesis, reference in zip(predictions, references):
            hypothesis = hypothesis.lower()
            reference = reference.lower()
            if self.remove_space:
                hypothesis = hypothesis.replace(" ", "")
                reference = reference.replace(" ", "")
            self.edits += editdistance.eval(hypothesis, reference)
            self.total += len(reference)

    def compute(self):
        return self.edits / max(1, self.total)

    def __call__(self, predictions, references):
        self.reset()
        self.update(predictions, references)
        return self.compute()
