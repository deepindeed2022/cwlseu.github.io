class Init(object):
    def __init__(self, value):
        self.val = value
        print "init", self.val

class Add2(Init):
    def __init__(self, val):
        super(Add2, self).__init__(val)
        print "Add2 Before:", self.val
        self.val += 2
        print "Add2", self.val

class Mul5(Init):
    def __init__(self, val):
        super(Mul5, self).__init__(val)
        print "Mul5 Before:", self.val
        self.val *= 5
        print "Mul5", self.val

class Pro(Mul5, Add2):
    pass

class Incr(Pro):
    def __init__(self, val):
        super(Pro, self).__init__(val)
        self.val += 1
        print "Incr", self.val

p = Incr(5)
print(p.val)