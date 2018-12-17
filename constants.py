import multiprocessing

nan = 10 ** 5
opp0 = 10**6

proc_num = multiprocessing.cpu_count()
if proc_num<=6:
    proc_num -= 1
else:
    proc_num -=2
