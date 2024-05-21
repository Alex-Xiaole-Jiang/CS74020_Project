from random import randrange
from random import seed
def shuffle_no_fixed_point(list, rand_seed = 37):
    seed(a = rand_seed)
    length = len(list)
    for i in range(length):
        randpos = randrange(i, length)
        list[i], list[randpos] = list[randpos], list[i]
    pass