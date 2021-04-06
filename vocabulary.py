import os

class Vocabulary:

    # this is where the words are stored
    content = []
    size = 0

    def load(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    self.content.append(line.strip('\n'))
                self.size = len(self.content)

    def getIndex(self, word):
        for i in range(0, len(self.content)):
            if self.content[i] == word:
                return i
        return None