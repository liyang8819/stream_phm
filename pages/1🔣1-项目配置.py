# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:30:20 2022

@author: LiYang
"""
import streamlit as st
import os
import json
from utils import get_pj_name,base_path
import shutil
import yaml
if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')
    
    
else:
    col1,col2=st.columns([2,3])
    col2.title('项目配置')   
    project_name=st.text_input("项目名称:")             
    project_desc=st.text_input("项目描述:")     
    project_save=st.button("确认") 
    
    if project_save:
        if project_name and project_desc:
            os.path.join(base_path, 'projects',project_name)
            os.mkdir(os.path.join(base_path, 'projects',project_name))
            os.mkdir(os.path.join(base_path, 'projects',project_name,'oridata'))
            os.mkdir(os.path.join(base_path, 'projects',project_name,'processed_data'))
            
    
            project_configs={}
            with open(os.path.join(base_path, 'projects',project_name,'pj_configs.json'),'w') as f:
                json.dump(project_configs,f)
                        
            with open(os.path.join(base_path, 'projects',project_name,'device_configs.yaml'),'w') as f:                             
                yaml.dump({'device_type':'滚动轴承'},f, allow_unicode=True)                                       
                
            pj_name={'pj_name':project_name,'project_desc':project_desc}  
            with open(os.path.join(base_path, 'pj_name.json'),'w') as f:
                json.dump(pj_name,f)                    
            st.success('项目创建成功，当前工作项目为'+project_name)
        else:
            st.warning("请填写完整项目信息")
        
    for i in range(3):
        st.markdown("") 
        
    projects_dir=os.path.join(base_path, 'projects')    
    if project_name in os.listdir(projects_dir):
        default_project_index=os.listdir(projects_dir).index(project_name)
    else:
        pj_name_last=get_pj_name()
        default_project_index=os.listdir(projects_dir).index(pj_name_last)
    project_sel=st.selectbox("切换项目",os.listdir(projects_dir),index=default_project_index) 
    
    project_sel_save=st.button('保存')
    if project_sel_save:       
        pj_name={'pj_name':project_sel}  
        with open('pj_name'+'.json','w') as f:
            json.dump(pj_name,f)                    
        st.success('项目已切换成'+project_sel)
        
        
    for i in range(3):
        st.markdown("")         
    project_del_name=st.selectbox("删除项目",os.listdir(projects_dir))     
    project_del=st.button("确认删除")     
    if project_del:
        if project_sel==project_del_name:
            st.warning("请先切换当前项目到非工作状态")
        else:
            shutil.rmtree(os.path.join(projects_dir, project_del_name))
                    
           
        


