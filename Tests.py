class yo:
    def __init__(self):
        print("Yo yo honey singh")
        self.d=25

class parent(yo):
    def __init__(self):
        yo.__init__(self)
        print("Howdy doWdy!")
        self.var=23


    def pr(self):
        print(self.d)


class child(parent, yo):


    def print(self):
        print(self.var, self.d)



c=child()
c.print()