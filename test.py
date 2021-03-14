
# def len_argsort(seq):
#     return sorted(range(len(seq)), key=lambda x: len(seq[x]))
#
# test = ['BOS', '我', '沒', '事', '。', 'EOS']
#
# res = len_argsort(test)
#
# print(res)
import math
import time

start_time = time.time()

for i in range(1000000000):
    res = math.atan(1.0)

end_time = time.time()

print(end_time - start_time)