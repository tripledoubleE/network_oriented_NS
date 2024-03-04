import os

def main():
    # Example lists of parameters
    list_param1 = ['BPR', 'LightGCN', 'NGCF', 'NNCF', 'SGL']
    list_param2 = ['a', 'b', 'c'] # her modelin parameter.json dosyasinin path'i
    list_param3 = [10, 20, 30] #her modelin sonucunun kaydedilecegi yerin path'i. Asagida ornek var:
    # /home/ece/Desktop/latest/GranuRec/experiments/graph_level_experiment/L1/BPR/no_ns
    #/home/ece/Desktop/latest/GranuRec/experiments/graph_level_experiment/L1/BPR/pw-1 

    # Using a for loop to run the script with different parameters
    for p1, p2, p3 in zip(list_param1, list_param2, list_param3):
        command = f"run_recbole.py {p1} {p2} {p3}"
        os.system(command)

if __name__ == "__main__":
    main()