#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def main():
    conf_path = 'conf'
    result_path = 'result'
    try:
        os.mkdir(conf_path)
        os.mkdir(result_path)
    except OSError:
        print("Required directories are created.")
        pass

    empty_sts = 0

    try:
        empty_sts = os.path.getsize('conf/.mapbox_token')
    except OSError:
        print('Token file exists.')
        pass

    if (empty_sts == 0):
        token = open('conf/.mapbox_token', 'w')
        token_key = input('Paste your Mapbox public token and press Enter: ')
        token.write(token_key)

    try:
        empty_sts = os.path.getsize('conf/.mapbox_token')
    except OSError:
        print('Token file exists.')
        pass

    if (empty_sts == 0):
        style = open('conf/.mapbox_style', 'w')
        style_key = input('Paste your Mapbox style URL and press Enter: ')
        style.write(style_key)

    print('Setup finished!')

if __name__ == "__main__":
    main()