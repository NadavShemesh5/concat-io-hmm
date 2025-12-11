class Batch:
    def __init__(self, messages):
        self.messages = messages

class Message:
    def __init__(self, content, prob):
        self.content = content
        self.prob = prob