import sys

file = open(str(sys.argv[1]),"r")
#lista = file.read().split(",")
lista = file.readlines()
print(lista)
nx = lista[1].index("[")
xx = lista[1].index("]")
xxx = list(map(int,lista[1][nx+1:xx].split(" ")))
print(xxx)
