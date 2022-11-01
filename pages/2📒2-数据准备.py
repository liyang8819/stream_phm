# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:20:45 2022

@author: LiYang
"""
import streamlit as st
import pandas as pd
from utils import get_pj_name,get_data_name,base_path
import os
import json

if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')
    
else:
   
    col1,col2=st.columns([2,3])  
    col2.title('数据准备')   
    tab1, tab2 = st.tabs(["🗃上传数据", "🗃连接数据库"])
    pj_name=get_pj_name()                    
        


    with tab1:    
        uploaded_file = st.file_uploader('上传数据（csv）',accept_multiple_files=False,type=['csv'])         
        dataupload_save=st.button("确认",key='uploadconfirm') 
        if dataupload_save and uploaded_file:
                               
            pj_name=get_pj_name()
            upload_frame=pd.read_csv(uploaded_file,index_col=None)
            projects_dir=os.path.join(base_path, 'projects')
            project_sel_dir=os.path.join(base_path, 'projects',pj_name) 
                                    
            upload_frame.to_csv(os.path.join(project_sel_dir, 'oridata',uploaded_file.name),index=None)
            upload_frame.to_csv(os.path.join(project_sel_dir, 'processed_data',uploaded_file.name),index=None)
            
            with open(os.path.join(project_sel_dir, 'pj_configs.json'),'r') as f:
                pj_configs=json.load(f)  
            pj_configs['data_name']=uploaded_file.name  
            pj_configs['data_source']='oridata'              
            with open(os.path.join(project_sel_dir, 'pj_configs.json'),'w') as f:
                json.dump(pj_configs,f) 
            
            st.success('数据上传成功,当前工作数据集为'+uploaded_file.name)                 
            upload_frame
            
        for i in range(3):
            st.markdown("") 
         
        try:
            data_name_last=get_data_name(pj_name) 
            pj_name_last=get_pj_name()
            projects_dir=os.path.join(base_path, 'projects')
            project_sel_dir=os.path.join(base_path, 'projects',pj_name) 
            
            data_dir=os.path.join(project_sel_dir, 'oridata')       
            default_data_index=os.listdir(data_dir).index(data_name_last) 
                       
            data_sel=st.selectbox("切换数据集",os.listdir(data_dir),index=default_data_index) 
            data_sel_save=st.button('保存')
            if data_sel_save:  
                
                with open(os.path.join(project_sel_dir, 'pj_configs.json'),'r') as f:
                    pj_configs=json.load(f)  
                pj_configs['data_name']=data_sel  
                with open(os.path.join(project_sel_dir, 'pj_configs.json'),'w') as f:
                    json.dump(pj_configs,f) 
                   
                st.success('已选择数据集'+data_sel)
        except:
            st.info("当前可用数据集为空，请先上传")
            
            
    with tab2:         
        source_ip=st.text_input("数据库地址")
        source_type=st.selectbox("数据库类型",['MongoDB','PostgreSQL','MySQL','Microsoft SQL Server','AWS S3'])              
        source_name=st.text_input("数据库名称")
        source_sheetname=st.text_input("表单名称")
        source_pointname=st.text_input("点位名称")
        
        dataconn_save=st.button("确认") 
        if dataconn_save:
            st.success('数据接口创建成功')
