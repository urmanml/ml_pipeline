
1,1,2,3,5, 8


list= [1,1,2,3,5, 8]
number =3
def func_fibonacci(number):
    list= []
    for i in range(0,number):

        if len(list)==0:
            list.append(1)
        elif len(list)==1:
            list.append(list[-1])

        elif len(list) > 1:
            list.append(list[-1]+list[-2])
    return list


func_fibonacci(2)


func_fibonacci(1)

func_fibonacci(5)
number=0
for i in range(0, number-1):
    print(i)

