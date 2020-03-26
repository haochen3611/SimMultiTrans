#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json


def main():
    conf_path = 'conf'
    result_path = 'result'
    try:
        os.mkdir(conf_path)
        os.mkdir(result_path)
    except OSError:
        print("Required directories are created.")
        pass

    print('Setting Mapbox...')
    empty_sts = 0

    try:
        empty_sts = os.path.getsize(f'{conf_path}/.mapbox_token')
    except OSError:
        print('Token file does not exist.')
        pass

    if (empty_sts == 0):
        token = open(f'{conf_path}/.mapbox_token', 'w')
        token_key = input('Paste your Mapbox public token and press Enter: ')
        token.write(token_key)

    try:
        empty_sts = os.path.getsize(f'{conf_path}/.mapbox_style')
    except OSError:
        print('Style file does not exist.')
        pass

    if (empty_sts == 0):
        style = open(f'{conf_path}/.mapbox_style', 'w')
        style_key = input('Paste your Mapbox style URL and press Enter: ')
        style.write(style_key)
    
    print('Complete!')

    print('Setting Input Data')
    try:
        empty_sts = os.path.getsize(f'{conf_path}/conf.json')
    except OSError:
        print('Configureation file does not exist.')
        pass

    if (empty_sts == 0):
        geo_path = input('Enter your geographic information file path (required): ')
        arr_rate_path = input('Enter your arrival rate file path (required): ')
        vehicle_attri_path = input('Enter your vehicle attribute file path (required): ')

    conf = {
        'geo_path': geo_path,
        'arr_rate_path': arr_rate_path,
        'vehicle_attri_path': vehicle_attri_path
    }

    with open(f'{conf_path}/conf.json', 'w') as json_file:
        json.dump(conf, json_file)

    print('Setup finished!')

if __name__ == "__main__":
    main()