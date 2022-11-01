# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:10:01 2022

@author: LiYang
"""

import streamlit as st
import yaml
import streamlit_authenticator as stauth

# from PIL import Image 
from utils import init_page,login,open_url


if __name__ == "__main__":
    
    init_page(st)
    authentication_status=login(st,stauth)     
            
        