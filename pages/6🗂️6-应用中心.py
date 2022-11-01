# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:30:20 2022

@author: LiYang
"""


import streamlit as st
import pandas as pd
import os
import pickle as pkl               
from utils import get_time_domain_features,get_freq_domain_features,base_path
if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')
    
else:
    col1,col2=st.columns([2,3])  
    col2.title('应用中心')  
    appdict_dir=os.path.join(base_path,'appdict') 

    tab1,tab2 = st.tabs(["🗃应用中心", 
                            "🗃实时监测"])    
    with tab1:
        col1,col2 =st.columns(2)
        with col1:
            appx=st.selectbox('选择应用',os.listdir(appdict_dir))
        if st.button('删除已选应用'):
            if os.path.exists(os.path.join(appdict_dir,appx)):
                os.remove(os.path.join(appdict_dir,appx))
            
        with open(os.path.join(appdict_dir,appx) ,'rb') as f:
            appdict=pkl.load(f)
        with st.expander('应用说明'):   
            st.markdown(appdict['app_desc'])   
        with st.expander('应用信息'):     
            appdict

    with tab2:
        import time
        from utils import time_fea_name
        realdata=appdict['dataframe']
        cols=st.selectbox('选择列', realdata.columns)
        
        paras=st.selectbox('选择参数', ['实时振动值']+list(appdict['config'].keys())+time_fea_name)                       
        limit=st.number_input('设定报警上限',value=15)
        monitor_period=st.number_input('单个采样段样本数',value=32768)
        calc_times=int(len(realdata)/monitor_period)
        st.write(calc_times,'个样本','每个样本包含',monitor_period,'个采样点')
        if paras in list(appdict['config'].keys()): 
            col5,col6=st.columns(2)
            with col5:
                st.write('')
                st.write('')
                st.write('')
                hilbert_c=st.checkbox('包络解调')
            with col6:
                n_freqbase=st.number_input('倍频系数', min_value=1,max_value=10)
                
        monitor_button=st.button('开始时序监控')        
        progress_bar = st.progress(0)
        
        
        ##时域特征监控
        if paras in time_fea_name:
            
            ##初始趋势图
            new_rows = realdata[[cols]][0:monitor_period].T    
            time_fea=pd.DataFrame(get_time_domain_features(new_rows),index=time_fea_name).T
            # limit_frame=pd.DataFrame([limit],index=[paras]).T
            chart = st.line_chart(time_fea[paras])
            # chart.add_rows(limit_frame)
            if monitor_button:
                ##时间趋势图
                for i in range(1,calc_times):
                    progress_bar.progress((i + 1)/calc_times)  
                    
                    new_rows = realdata[[cols]][i*monitor_period:(i+1)*monitor_period].T                    
                    time_fea_new=pd.DataFrame(get_time_domain_features(new_rows),index=time_fea_name).T

                    chart.add_rows(time_fea_new[paras])
                    # chart.add_rows(limit_frame)
                    if time_fea_new[paras].max()>limit:
                        st.warning(str(i)+'---'+cols+'---'+paras+'超限,限值='+str(limit))
                    time.sleep(0.1)
                    
         ##频域特征监控     
                   
        if paras in list(appdict['config'].keys()): 
            new_rows = realdata[[cols]][0:monitor_period]   
            freq_amp=get_freq_domain_features(new_rows,hilbert_c,sample_rate=appdict['data_freq'])
            freqbase=appdict['config'][paras]*n_freqbase
            amp=freq_amp.loc[freqbase-appdict['freqbase_error']:freqbase+appdict['freqbase_error']].max().values[0]
  
            chart = st.line_chart(pd.DataFrame([amp]))
            
            if monitor_button:
                for i in range(1,calc_times):
                    progress_bar.progress((i + 1)/calc_times)  
                    new_rows = realdata[[cols]][i*monitor_period:(i+1)*monitor_period]                   
                    freq_amp=get_freq_domain_features(new_rows,hilbert_c,sample_rate=appdict['data_freq'])
          
                    amp=freq_amp.loc[freqbase-appdict['freqbase_error']:freqbase+appdict['freqbase_error']].max().values[0]
                    chart.add_rows(pd.DataFrame([amp]))
                #     # chart.add_rows(limit_frame)
                    if amp>limit:
                        st.warning(str(i)+'---'+cols+'---'+paras+'超限,限值='+str(limit))
                    time.sleep(0.1)
                
                st.success('趋势监控完毕')
        if paras =='实时振动值':
            new_rows = realdata[[cols]][0:int(monitor_period)] 
            chart = st.line_chart(new_rows)
            if monitor_button:
                for i in range(1,calc_times):
                    progress_bar.progress((i + 1)/calc_times)  
                    new_rows = realdata[[cols]][i*monitor_period:(i+1)*monitor_period]                   
    
                    chart.add_rows(new_rows)
                #     # chart.add_rows(limit_frame)
                    # if np.max(new_rows)>limit:
                    #     st.warning(str(i)+'---'+cols+'---'+paras+'超限,限值='+str(limit))
                    time.sleep(0.1)
                
                st.success('趋势监控完毕')
            
                      