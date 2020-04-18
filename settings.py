'''
for storing arguments
'''

main_args=dict()
main_args['batch_size'] = 128
main_args['budget'] = 30
main_args['split'] = 1
main_args['name'] = 'self-david' #'skeleton'#'self-vanilla'

def get_dict():
    return main_args

def get(key):
    if key in main_args.keys():
        return main_args[key]
    else:
        raise AttributeError(f'Invalid key here are the options: {main_args}')