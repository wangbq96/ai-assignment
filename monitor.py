import psutil
import sys
import csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please enter pid")
    else:
        cpu_load_list = []
        men_load_list = []
        try:
            pid = int(sys.argv[1])
            p = psutil.Process(pid)
            while True:
                cpu_load = p.cpu_percent(interval=1) / psutil.cpu_count()
                mem_load = p.memory_percent()
                cpu_load_list.append(cpu_load)
                men_load_list.append(mem_load)
                print("CPU load:", cpu_load, "Memory load:", mem_load)
        except psutil.NoSuchProcess:
            print('awsl')
            with open("monitor.csv", "w", newline="") as f:
                f_csv = csv.writer(f)
                headers = ["cpu_load", "mem_load"]
                f_csv.writerow(headers)
                for i in range(len(men_load_list)):
                    row = [cpu_load_list[i], men_load_list[i]]
                    f_csv.writerow(row)



