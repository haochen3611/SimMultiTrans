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
        #print ("Creation of the directory %s failed")
        pass

    token = open('conf/.mapbox_token', 'w')
    token_key = input('Paste your Mapbox public token and press Enter: ')
    token.write(token_key)

    style = open('conf/.mapbox_style', 'w')
    style_key = input('Paste your Mapbox style URL and press Enter: ')
    style.write(style_key)

    print('Done!')

if __name__ == "__main__":
    main()