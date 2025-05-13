def isValid(s):
    listChars = {}
    for i in s:
        if i in listChars:
            listChars[i] = listChars[i]+1
        else:
            listChars[i]=1
    odd = 0
    for i in listChars:
        if listChars[i]%2==1:
            odd += 1
    if odd>1:
        return False
    else:
        return True
