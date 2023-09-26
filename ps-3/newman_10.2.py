import random
import matplotlib.pyplot as plt

num_213Bi = 10000
num_209Tl = 0
num_209Pb = 0
num_209Bi = 0

prob_213Bi_to_209Tl = 0.0209
prob_213Bi_to_209Pb = 0.9791
half_life_209Tl = 2.2 * 60  
half_life_209Pb = 3.3 * 60  
half_life_213Bi = 46 * 60

time_values = []
count_213Bi = []
count_209Tl = []
count_209Pb = []
count_209Bi = []

total_time = 20000
delta_t = 1  

for time in range(total_time + 1):
    time_values.append(time)
    count_213Bi.append(num_213Bi)
    count_209Tl.append(num_209Tl)
    count_209Pb.append(num_209Pb)
    count_209Bi.append(num_209Bi)

    num_decayed_209Tl = 0
    if num_209Tl > 0:
        for _ in range(num_209Tl):
            if random.random() < (1 - 2 ** (-delta_t / half_life_209Tl)):
                num_decayed_209Tl += 1
    num_209Tl -= num_decayed_209Tl
    num_209Pb += num_decayed_209Tl

    num_decayed_209Pb = 0
    if num_209Pb > 0:
        for _ in range(num_209Pb):
            if random.random() < (1 - 2 ** (-delta_t / half_life_209Pb)):
                num_decayed_209Pb += 1
    num_209Pb -= num_decayed_209Pb
    num_209Bi += num_decayed_209Pb

    num_decayed_213Bi_to_209Tl = 0
    num_decayed_213Bi_to_209Pb = 0
    if num_213Bi > 0:
        for _ in range(num_213Bi):
            if random.random() < (1 - 2 ** (-delta_t / half_life_213Bi)):
                if random.random() < prob_213Bi_to_209Tl:
                    num_decayed_213Bi_to_209Tl += 1
                else:
                    num_decayed_213Bi_to_209Pb += 1
    num_213Bi -= num_decayed_213Bi_to_209Tl + num_decayed_213Bi_to_209Pb
    num_209Tl += num_decayed_213Bi_to_209Tl
    num_209Pb += num_decayed_213Bi_to_209Pb

plt.figure(figsize=(10, 6))
plt.plot(time_values, count_213Bi, label='213Bi')
plt.plot(time_values, count_209Tl, label='209Tl')
plt.plot(time_values, count_209Pb, label='209Pb')
plt.plot(time_values, count_209Bi, label='209Bi')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Atoms')
plt.legend()
plt.grid(True)
plt.show()
