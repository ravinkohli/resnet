'''
for storing arguments
'''

main_args=dict()
main_args['batch_size'] = 8
main_args['budget'] = 30
main_args['split'] = 0.8
main_args['name'] = 'skeleton'

def get_dict():
    return main_args
def get(key):
    if key in main_args.keys():
        return main_args[key]
    else:
        raise AttributeError(f'Invalid key here are the options: {main_args}')