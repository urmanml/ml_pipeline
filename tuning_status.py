

tune_log_path= r'temporary_objects/tune_log.txt'

def tuning_status(tune_log_path):

    f = open(tune_log_path, "r")

    temp1 = f.readlines()[-1]

    def getNumbers(str):
        array = re.findall(r'[0-9.]*[0-9]+', str)
        return array

    temp2= getNumbers(temp1)
    elapsed= temp2[1]
    total= temp2[2]

    status= round(int(elapsed)/int(total)*100,2)


    return status

tuning_status(tune_log_path)


