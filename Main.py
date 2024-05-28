import time
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
from Desision_class import DesisionClass


directory_path = '/home/pouria/project/csv_files_for_main_program/'
directory_path_models = '/home/pouria/project/trained_models/'
directory_path_answer = '/home/pouria/project/answers/'
extension = '.csv'
extension_answer = '.txt'
extension_acn = '.acn'
extension_model = '_for_train.pkl'


def time_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    time_left = next_hour - now
    return time_left.total_seconds()


def delete_acns():
    files = os.listdir(directory_path_answer)
    if len(files) == 11 :
         time.sleep(1)
         for count in range (11) :
            temp = files[count]
            print(temp[-12:])
            if temp[-3:] == "acn" :
                os.remove(directory_path_answer+files[count])
                print(f" Deleting ACN files : {count}")

def delete_TXT_files():
    files = os.listdir(directory_path_answer)
    print(f"Number of file remaining if ANSWER directory is : {len(files)}")
    files1 = os.listdir(directory_path_answer)

    if len(files) == 22  :
        time.sleep(1)
        while len(files1) >= 1 :
            os.remove(directory_path_answer+files[len(files1)-1])
            # time.sleep(2)
            files1 = os.listdir(directory_path_answer)
            print(f" Deleting ACN and TXT files {len(files1)}")


while True:
    files = os.listdir(directory_path)
    print(f" Number of CSV files wating for making answer is :  {len(files)}")
    if len(files) == 11 :
            
        files_with_creation_time = []
        for filename in os.listdir(directory_path):
                if filename.endswith(extension):
                    file_path = os.path.join(directory_path, filename)
                    creation_time = os.path.getctime(file_path)
                    files_with_creation_time.append((file_path, creation_time))
        files_with_creation_time.sort(key=lambda x: x[1])
        for file_path, creation_time in files_with_creation_time:
            # file_path_answer = os.path.splitext(file_path)[0] + extension_answer
            # file_path_model = file_path[:-21] + 'TRAINED_MODELS/' +file_path[-12:-4] + extension_model
            Answer = [0,0]
            position = "NAN"          
            try :
                with open(f"/home/pouria/project/trained_models/{file_path[-12:-4]}_parameters.pkl", 'rb') as f:
                    ind = pickle.load(f)
                
                if ind[0] != 0 and ind[1] != 0 and ind[2] != 0 :
                    print(f"Start to predicting {file_path[-12:-6]} :")
                    print(f"reading {file_path[-12:]}")
                    data = pd.read_csv(file_path)
                    print("Runing Desision class for forcasting ...")
                    position, Answer = DesisionClass().Desision(data ,file_path[-12:-6] ,ind[:-1], ind[-1]  )
                    print(f"Position = {position}")
                    print(f"Answer = {Answer}")
                    print("remove csv file")
                    print("-----------------------------------")
                    os.remove(file_path)
                    with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
                        file.write(position+"\n")
                        file.write(str(Answer[0])+"\n")
                        file.write(str(Answer[1]))
                else :
                    print(f"There is no saved model for {file_path[-12:-6]} so delete csv file")
                    os.remove(file_path)
                    with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
                        file.write(position+"\n")
                        file.write(str(Answer[0])+"\n")
                        file.write(str(Answer[1]))
            
            except :
                print(f"Error in reading some files related to : {file_path[-12:-6]} so delete csv file")
                with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
                    file.write(position+"\n")
                    file.write(str(Answer[0])+"\n")
                    file.write(str(Answer[1]))
                os.remove(file_path)


            # file_path_akn = file_path[:-3] + "akn"
            # with open(file_path_akn, 'w') as file:
            #     file.write('aknowladgement')

            # position, Answer = DesisionClass().Desision(data ,file_path_model ,page = 2 ,feature = 30 ,QTY = 1000 ,depth = 2  )

            # print(file_path[-12:-6]+" ------> "+position)

            # with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
            #     file.write(position+"\n")
            #     file.write(str(Answer[0, 0])+"\n")
            #     file.write(str(Answer[0, 1]))

            # os.remove(file_path)



              
    time.sleep(1)
    delete_acns()

    remaining_time = time_until_next_hour()
    print(f"Main Expert showing you the progress of next job ...")
    with tqdm(total=remaining_time, desc="Time until next hour", unit="s") as pbar:
        while remaining_time > 0 :
            files = os.listdir(directory_path_answer)
            if len(files) == 11 :   
                print("New files for calculating just recived progress broken to coninue proccecing")
                delete_acns()
                break
            files = os.listdir(directory_path_answer)            
            if len(files) == 22 :   
                print("New files for calculating just recived progress broken to coninue proccecing")
                delete_TXT_files()
                break     
            
            files = os.listdir(directory_path)
            if len(files) == 11 :   
                break       
            time.sleep(1)
            remaining_time -= 1
            pbar.update(1)
            files = os.listdir(directory_path_answer)
            print(f"Number of file remaining if ANSWER directory is : {len(files)}")

    print("Next hour has started!")
