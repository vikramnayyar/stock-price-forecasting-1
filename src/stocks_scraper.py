#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scrip downlads the stock data from yahoo finance
"""

import yfinance as yf

df = yf.download('CADILAHC.NS',
                        start = '2000-09-1',   # 2000-09-19
                        end = '2021-06-30', 
                interval = '1d')

df.to_csv('../data/cadilahc.csv')