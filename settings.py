'''
for storing arguments
'''

main_args=dict()

def get(key):
    if key in main_args.keys():
        return main_args[key]
    else:
        raise AttributeError(f'Invalid key here are the options: {main_args}')