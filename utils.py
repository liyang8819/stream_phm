# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:21:00 2022

@author: LiYang
"""
import streamlit as st

import yaml
import pickle
import webbrowser
from PIL import Image
import json
import pandas as pd
import seaborn as sns
import os
import numpy as np
from scipy.fftpack import fft            
from scipy.signal import hilbert, chirp
from bokeh.plotting import figure   
from bokeh.models.ranges import Range1d  
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Label
import math
time_fea_name=['均值','绝对平均值','方差','标准差','方根幅值','均方根值','峰值','最大值','最小值',
   '波形指标','峰值指标','脉冲指标','裕度指标','偏斜度','峭度']

base_path=os.path.dirname(os.path.abspath('home.py'))

def init_page(st):
    st.set_page_config(page_title="HCE-idas故障诊断平台", 
                        page_icon="random" , 
                        # layout="wide",
                        initial_sidebar_state="auto")
    #隐藏脚注
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: visiable;}
                footer {visibility: hidden;}
                </style>               
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    
    #设置markdown字体，方便引用    
    st.markdown(""" <style> .font3 {
    font-size:30px ; font-family: 'Cooper Black'; color: red;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:40px ; font-family: 'Cooper Black'; color: red;} 
    </style> """, unsafe_allow_html=True)   
    
    #设置界面logo    
    image = Image.open('logo/wave3.jpg') 
    st.image(image, use_column_width=True) 
    st.markdown('<p class="font1">欢迎使用idas故障诊断平台</p>', unsafe_allow_html=True)

    
    
    
def login(st,stauth):    
    with open('credentials.yaml') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status:
           
        col1,col2,col3=st.columns([3,1,1])
        with col2:
            st.write(f'当前用户:*{name}*')
        with col3:     
            authenticator.logout('Logout', 'main')
        
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    return authentication_status

def open_url(urls):
    
    webbrowser.open(urls, new=2, autoraise=False)
    
def get_pj_name():
    
    with open(os.path.join(base_path, 'pj_name.json') ,'r') as f:
        pj_name=json.load(f)['pj_name']  
    return pj_name

def get_data_name(pj_name):
    
    with open(os.path.join(base_path,'projects',pj_name, 'pj_configs.json') ,'r') as f:
        data_name=json.load(f)['data_name']  
    return data_name

def get_data_source(pj_name):
    with open(os.path.join(base_path,'projects',pj_name, 'pj_configs.json'),'r') as f:
        data_source=json.load(f)['data_source']  
    return data_source


# @st.cache(suppress_st_warning=True)
def data_read_cache(data_name,pj_name,data_source):
    """将最近一次上传的数据进行缓存"""       
    data=pd.read_csv(os.path.join(base_path,'projects',pj_name, data_source,data_name),index_col=0).convert_dtypes()      
    return data  



def corr_plot(dataframe):
    plot_data = dataframe.replace({np.inf: np.nan, -np.inf: np.nan}) # 无穷大和无穷小替换为nan
    plot_data = plot_data.dropna()
    sns.pairplot(plot_data,kind='reg',diag_kind='kde')
      

def save_models(best_models,train_results,train_results_save=True): 
    with open ('best_models.pkl','wb') as f:
        pickle.dump(best_models,f)
    if train_results_save:
        with open ('train_results.pkl','wb') as f:
            pickle.dump(train_results,f) 
            


def envelope_demodulation_v1(data,fs):
    """
    envelope demodulation for a time series signal

    Input parameters:
    -----------------
    data      : input data, Series
    fs        : sampling frequency """
        
    # data = data.values.reshape(-1) #data需要为一维的array或list
    L=len(data)   
    Y =fft(np.abs(hilbert(data))-np.mean(np.abs(hilbert(data))))
    freq=np.arange(int(L/2))*fs/L  #频率范围0-fs/2，频率分辨率=采样数据长度/2
    amp=np.abs(Y)[0:int(L/2)]*2/L #幅值处理
    
    p = figure(
    title='FFT result',
    x_axis_label='频率(Hz)',
    y_axis_label='频率幅值',
    width=1000, height=300, background_fill_color="black")
    p.y_range = Range1d(0, 0.1)
    p.line(freq.tolist(),amp.tolist(),color='red')
    p.circle(freq.tolist(),amp.tolist(),color='red') 
    
    # p.yaxis.bounds =(0,max((2*p1/L).tolist())) 
    
    st.bokeh_chart(p) 
    
    return pd.DataFrame([freq,amp],index=['f','amp']).T.sort_values(by='amp',ascending=False)


       
def fft_plot(data,sample_rate,st,hilbert_=False,diag_det=False,diag_para=None,diag_sel=None,fft_plot_max_freq=None):
   
    L=len(data)
    Y = fft(data) if not hilbert_ else fft(np.abs(hilbert(data))-np.mean(np.abs(hilbert(data))))
    amp = 2*np.abs(Y)[:int(L/2)]/L # 单侧频谱
    fbin = np.arange(int(L/2))*sample_rate/L;  #0-500hz，L/2个频率点   
    
    if fft_plot_max_freq:
        amp=amp[0:int(fft_plot_max_freq)]
        fbin=fbin[0:int(fft_plot_max_freq)]
    
    p = figure(
                title='FFT result',
                x_axis_label='频率(Hz)',
                y_axis_label='频率幅值',
                width=1000, 
                height=300, 
                background_fill_color="black")
    
    p.y_range = Range1d(0,np.max(amp)+1)
    p.line(fbin.tolist(),amp.tolist(),color='yellow')
    p.circle(fbin.tolist(),amp.tolist(),color='yellow')

    
    
    #计算故障频率
    freq_amp=pd.DataFrame([fbin,amp],index=['f','amp']).T
    freq_amp_sorted=freq_amp.sort_values(by='amp',ascending=False)

    if diag_det:
        for dia_name in diag_sel:
            freq_fault_low=diag_para[dia_name]['freq_fault_low'] 
            freq_fault_high=diag_para[dia_name]['freq_fault_high'] 

            fault_amp=freq_amp[freq_amp['f']>=freq_fault_low]
            fault_amp=fault_amp[fault_amp['f']<=freq_fault_high]
            with st.expander(dia_name+' 故障误差区间：'):
                st.table(fault_amp)
            if len (fault_amp):
                freq_fault=fault_amp[fault_amp['amp']==fault_amp['amp'].max()]
                st.write(dia_name+'-故障信息：',freq_fault)
                source = ColumnDataSource(
                                        data=dict(
                                        f=[freq_fault['f'].values[0]],
                                        amp=[freq_fault['amp'].values[0]],
                                        names=[dia_name]))
                    
                p.scatter(
                        x='f',
                        y='amp',
                        size=5,
                        color='red',
                        source=source
                    )
            
                
                label = Label(
                            x=freq_fault['f'].values[0],
                            y=freq_fault['amp'].values[0],
                            x_offset=0,
                            y_offset=5,
                            text=dia_name,
                            text_baseline="middle",
                            text_color='red')
                        
                p.add_layout(label)
    
    st.bokeh_chart(p)
    return freq_amp_sorted  


def signalt_generate(sample_rate,data_len):
    return np.arange(data_len)/sample_rate  

          
def get_time_domain_features(data):
    '''data为一维振动信号'''\
        
    x_rms = 0
    absXbar = 0
    x_r = 0
    S = 0
    K = 0
    
    fea = []
    len_ = len(data.iloc[0, :])
    mean_ = data.mean(axis=1).values  # 1.均值
    var_ = data.var(axis=1).values  # 2.方差
    std_ = data.std(axis=1).values  # 3.标准差
    max_ = data.max(axis=1).values  # 4.最大值
    min_ = data.min(axis=1).values  # 5.最小值
    x_p = max(abs(max_[0]), abs(min_[0]))  # 6.峰值
    
    
    for i in range(len_):
        x_rms += data.iloc[0, i] ** 2
        absXbar += abs(data.iloc[0, i])
        x_r += math.sqrt(abs(data.iloc[0, i]))
        S += (data.iloc[0, i] - mean_[0]) ** 3
        K += (data.iloc[0, i] - mean_[0]) ** 4
        
        
    x_rms = math.sqrt(x_rms / len_)  # 7.均方根值
    absXbar = absXbar / len_  # 8.绝对平均值
    x_r = (x_r / len_) ** 2  # 9.方根幅值
    W = x_rms / mean_[0]  # 10.波形指标
    C = x_p / x_rms  # 11.峰值指标
    I = x_p / mean_[0]  # 12.脉冲指标
    L = x_p / x_r  # 13.裕度指标
    S = S / ((len_ - 1) * std_[0] ** 3)  # 14.偏斜度
    K = K / ((len_ - 1) * std_[0] ** 4)  # 15.峭度

    fea = [mean_[0],absXbar,var_[0],std_[0],x_r,x_rms,x_p,max_[0],min_[0],W,C,I,L,S,K]
    return fea

def get_freq_domain_features(data,hilbert_,sample_rate):
    
    
    L=len(data)
    Y = fft(data) if not hilbert_ else fft(np.abs(hilbert(data))-np.mean(np.abs(hilbert(data))))
    amp = 2*np.abs(Y)[:int(L/2)]/L # 单侧频谱
    fbin = np.arange(int(L/2))*sample_rate/L;  #0-500hz，L/2个频率点   
    freq_amp=pd.DataFrame(amp,index=fbin,columns=['amp'])
    
    return freq_amp

