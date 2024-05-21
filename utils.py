from random import randrange
from random import seed
def shuffle_no_fixed_point(list, rand_seed = 37):
    seed(a = rand_seed)
    new_list = list
    length = len(new_list)
    for i in range(length):
        randpos = randrange(i, length)
        new_list[i], new_list[randpos] = new_list[randpos], new_list[i]
    return new_list