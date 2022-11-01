# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:30:20 2022

@author: LiYang
"""
import streamlit as st
import os
import json
import yaml
import pandas as pd
import pickle as pkl
from utils import base_path
import shutil
if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')

    
else:
    col1,col2=st.columns([2,3])
    col2.title('创建应用')   
    app_name=st.text_input("1，应用名称")  
    with st.expander('2,应用描述'):           
        app_desc=st.text_input("") 
    uploaded_data = st.file_uploader('3,上传数据（csv）',accept_multiple_files=False,type=['csv'])   
    uploaded_config= st.file_uploader('4,上传故障频率配置文件',type=['yaml'])        
    data_freq=st.number_input('5,数采频率') 
    freqbase_error=st.number_input('6,故障频率偏差',key='频率偏差') 
    
    appdict={}
    appdict['app_name']=app_name
    appdict['dataframe']=pd.read_csv(uploaded_data) if uploaded_data else None 
    
    appdict['app_desc']=app_desc 
    
    # with open('projects/'+pj_name+'/'+'fault_freq.yaml') as f:                             
    appdict['config']=yaml.load(uploaded_config, Loader=yaml.SafeLoader) if uploaded_config else None 
    # appdict['config']=uploaded_config                    
    appdict['data_freq']=data_freq
    appdict['freqbase_error']=freqbase_error 
    create=st.button('创建')
    if create:
        if app_name !=None and uploaded_data != None and uploaded_config != None and data_freq!= None and freqbase_error!= None:
            
            appdict_dir=os.path.join(base_path,'appdict',app_name) 
            with open(appdict_dir,'wb') as f:
                pkl.dump(appdict,f)
            st.info(app_name+"应用已创建")
        else:

            st.warning('请上传文件')
        
    

        
        


