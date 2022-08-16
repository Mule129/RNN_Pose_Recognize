import random as rd

def dice():
    """
    return : 랜덤 값(1~6) 반환
    """
    value = rd.randint(1,6)
    print(f"value : {value}")
    return value

def image(n):
    """
    매개변수 n : 정수를 입력받아 주사위 모양 아스키코드로 출력
    return : 주사위 모양 str 값 반환
    """
    up_end = " -----\n"
    data = ""
    print(f"n : {n}")
    if n%2 == 0:
        for _ in range(int(n/2)):
            data += "| 0 0 |\n"
    else:
        for _ in range(int((n-1)/2)):
            data += "| 0 0 |\n"
        data += "|  0  |\n"
    value = up_end + data + up_end

    return value

_ = input("주사위 굴리기 : 아무 키나 입력해주세요")

system = dice()
user = dice()
print(image(system))
print(image(user))

if system > user:
    print("system win!")
elif system == user:
    print("")
else:
    print("User win!")