# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:30:20 2022

@author: LiYang
"""


import streamlit as st
import shutil
from utils import get_pj_name,get_data_name,data_read_cache,base_path
import os


if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')
    
else:
    col1,col2=st.columns([2,3])  
    col2.title('数据处理')  
    # choose = stx.stepper_bar(steps=["数据预处理", "数据归一化", "特征降维", "特征构建", "特征筛选"], 
    #                          is_vertical=0, lock_sequence=False)
    
    tab1, tab2= st.tabs(["🗃数据处理","撤销所有对数据的处理"])    
    
    pj_name=get_pj_name()  
    data_name=get_data_name(pj_name)
    # data_source=get_data_source(pj_name) 
    data_source='processed_data'
    dataframe=data_read_cache(data_name,pj_name,data_source) 
    processed_data_dir=os.path.join(base_path,'projects',pj_name,'processed_data')
    processed_data_data_name_dir=os.path.join(base_path,'projects',pj_name,'processed_data',data_name)    
    
    with tab1:
        preprocess_choose=st.selectbox('处理方法',("设定索引", 
                                                   "数据筛选", 
                                                   "删除重复行"
                                                   ))
        
        if preprocess_choose=="设定索引":
            index_sel=st.selectbox('选择列',dataframe.columns)
            index_manual=st.checkbox('设定数字顺序索引')            
            save_index=st.button('确认')
            
            if save_index and index_manual:
                dataframe.index=range(len(dataframe))
                dataframe.to_csv(processed_data_data_name_dir,index=True)
            if save_index and index_sel and not index_manual:
                # dataframe.index=dataframe[index_sel]
                dataframe.set_index(index_sel,inplace=True)
                dataframe.to_csv(processed_data_data_name_dir,index=True)
            dataframe
            
        if preprocess_choose=="数据筛选":
            index_col_sel=st.multiselect('1，选择删除列',dataframe.columns)
            del_cols=st.button('确认',key='del_cols') 
            if del_cols:
                dataframe.drop(index_col_sel,axis=1,inplace=True)
                dataframe.to_csv(processed_data_data_name_dir,index=True)            
            
            index_row_sel=st.multiselect('2，选择删除行',dataframe.index)
            del_rows=st.button('确认',key='del_rows') 
            if del_rows:
                dataframe.drop(index_row_sel,axis=0,inplace=True)
                dataframe.to_csv(processed_data_data_name_dir,index=True)
                
                
            col1,col2,col3,col4=st.columns([1,1,1,1])
            index_col_btw=col1.selectbox('3，选择列数值范围',dataframe.columns)
            low=col2.number_input('最小')
            high=col3.number_input('最大')
            equ=col4.text_input('等于')
            btw_cols=st.button('确认',key='btw_cols')  
            if btw_cols and not equ:
                dataframe=dataframe[dataframe[index_col_btw]>low]
                dataframe=dataframe[dataframe[index_col_btw]<high]
                dataframe.to_csv(processed_data_data_name_dir,index=True) 
            if btw_cols and equ:
                dataframe=dataframe[dataframe[index_col_btw]>low]
                dataframe[dataframe[index_col_btw]==equ].to_csv(processed_data_data_name_dir,index=True)                 
               
            '4，设定索引范围'
            col1,col2,col3=st.columns([1,2,1])
            lowindex=col1.text_input('from',key='indexlow',value=dataframe.index[0])
            highindex=col3.text_input('to',key='indexhigh',value=dataframe.index[-1])
            idnex_btw_cols=st.button('确认',key='idnex_btw_cols') 
            if idnex_btw_cols:
                dataframe[int(lowindex):int(highindex)].to_csv(processed_data_data_name_dir,index=True) 
                dataframe    
        if preprocess_choose=="删除重复行":             
             ignore_index=st.checkbox('重置索引')                                  
             del_rows=st.button('确认',key='del_rows')
             if del_rows:
                 dataframe.drop_duplicates(inplace=True,ignore_index=ignore_index)  
                 dataframe.to_csv(processed_data_data_name_dir,index=True) 

             dataframe
             
    with tab2:
        resetdata=st.button('撤销所有对数据的处理',key='recovery')
        if resetdata:
            shutil.copyfile(os.path.join(base_path,'projects',pj_name,'oridata',data_name) , processed_data_data_name_dir)
        


