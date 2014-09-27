import createDataVoca as dv

# filetype = 'Adj'
# pattern = 'JJ.*'

# filetype = 'NN'
# pattern = 'NN.*'

filetype = 'AllWords'
pattern = 'AllWords'

data = dv.extractRevsLabel(filetype, pattern)