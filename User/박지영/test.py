class test():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def test_def(self):
        
        self.a
        print(self.a, self.b, self.c)

test_class = test(1, 20, 30)
test_class.test_def()