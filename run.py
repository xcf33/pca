from reduce import pca_experiment
from sims_test import sims_test


'''
Step 1, generate a reduced feature set dump
'''
pca = pca_experiment('8_changs-noodle-shop_encodings_vgg19_full.txt', 0.99)

'''
Step 2, generate a cache file for siggy based on the dump
'''
reduced = pca.decompose('transform')
print(reduced.shape)

'''
Step 3, generate a SIMs list JSON from both original data cache and reduced data cache
'''
# test = sims_test('reduced_8_changs-noodle-shop_encodings_vgg19.txt')
# meh = test.calc_sims()

print('done')