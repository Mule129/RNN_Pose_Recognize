from operator import index
from random import *

list = ['가위', '바위', '보']

computerN = randint(0,2)

user = input("가위바위보 게임 : ")

userN = list.index(user)
computer = list[computerN]
print("컴퓨터 : "+computer)

if computerN == userN :
    print("비김")
elif computerN<userN :
    print("이김")
else :
    print("짐")