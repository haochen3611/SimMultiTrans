#!/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    token = open('conf/.mapbox_token', 'w')
    token_key = input('Paste your Mapbox public token and press Enter: ')
    token.write(token_key)

    style = open('conf/.mapbox_style', 'w')
    style_key = input('Paste your Mapbox style URL and press Enter: ')
    style.write(style_key)

    print('Done!')

if __name__ == "__main__":
    main()