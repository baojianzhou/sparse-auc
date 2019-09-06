# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.rcParams.update({'font.size': 12})

names = ['QuickSelect-V1', 'QuickSelect-V2', 'MaxHeap-V1', 'Floyd-Rivest-V1', 'QuickSelect-V3',
         'Wirth', 'QuickSelect-V4', 'Floyd-Rivest-V2', 'QuickSort']
colors = ['b', 'b', 'm', 'green', 'b', 'y', 'b', 'green', 'r']
alpha_list = [1., .8, 1., 1., .6, 1., .4, .5, 1.]
run_time = {1: np.asarray(
    [621, 416, 977.0, 164, 403, 440, 549, 214, 9716]),
    10: np.asarray([5023, 4497, 9906.0, 1279, 3939, 3641, 3854, 1895, 95396]),
    100: np.asarray([46165, 38435, 98872.0, 12727, 38959, 36181, 38824, 19763, 956011]),
    1000: np.asarray([462480, 387161, 990644.0, 125644, 383959, 356307, 390857, 196206, 9606209]),
    2000: np.asarray(
        [973674, 799438, 2076245.0, 266914, 813301, 745576, 804968, 410895, 20250160]),
    5000: np.asarray(
        [2496992, 2028612, 5242117.0, 684493, 2058850, 1871607, 2055310, 1049421, 51434902]),
    8000: np.asarray(
        [3862229, 3189205, 8228015.0, 1058340, 3214370, 2938456, 3207977, 1640777, 80098740]),
    10000: np.asarray([4662248, 3909561, 10003924.0, 1271100, 3909567, 3583082, 3919146,
                       1989976, 97095163]),
    100000: np.asarray(
        [48050835, 39885553, 102413292.0, 13181102, 40028054, 36642196, 39946229, 20375441,
         999201785])}
x = [1, 10, 100, 1000, 2000, 5000, 8000, 10000]
fig, ax = plt.subplots(1, 2)
for ind, name in enumerate(names):
    vals = np.asarray([run_time[_][ind] for _ in x],
                      dtype=float) / 1e6
    ax[0].plot(x, vals, linewidth=2., label=name, color=colors[ind], alpha=alpha_list[ind],
               marker='.')
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('run time (seconds)')
    ax[0].set_title('Total run time w.r.t iterations')
    ax[0].grid(True)
    ax[0].legend()
for ind, name in enumerate(names):
    if name == 'QuickSort':
        continue
    vals = np.asarray([run_time[_][ind] for _ in x],
                      dtype=float) / 1e6
    ax[1].plot(x, vals, linewidth=2., label=name, color=colors[ind], alpha=alpha_list[ind],
             marker='.')
    ax[1].set_xlabel('iterations')
    ax[1].set_ylabel('run time (seconds)')
    ax[1].set_title('Total run time w.r.t iterations')
    ax[1].grid(True)
    ax[1].legend()
plt.subplots_adjust(wspace=0.5, hspace=0.2)
plt.savefig('/home/baojian/ht_run_time.png', dpi=400, bbox_inches='tight', pad_inches=0,
            format='png')
plt.show()
