# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:08:51 2022

@author: LiYang
"""
import streamlit as st
import pandas as pd
from utils import get_pj_name,get_data_name,data_read_cache,get_data_source,fft_plot,signalt_generate,get_time_domain_features,get_freq_domain_features,base_path
from bokeh.plotting import figure
import numpy as np
from scipy.signal import hilbert, chirp
import pywt
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
projects_dir=os.path.join(base_path, 'projects')


if not st.session_state["authentication_status"]:
    st.warning('请先在Home页面登录')
else:
    col1,col2=st.columns([2,3])  
    col2.title('数据探索')  
    if st.checkbox('切换数据集类型'):   
        datatype=st.selectbox('切换数据集类型', ['原始数据','已处理数据'])
        datatype_save=st.button('确认')
        if datatype=='原始数据' and datatype_save:
            pj_name=get_pj_name() 
            project_sel_dir=os.path.join(base_path, 'projects',pj_name) 
            project_sel_pj_configs_dir=os.path.join(project_sel_dir, 'pj_configs.json')
            
            with open(project_sel_pj_configs_dir,'r') as f:
                pj_configs=json.load(f)  
            pj_configs['data_source']='oridata'              
            with open(project_sel_pj_configs_dir,'w') as f:
                json.dump(pj_configs,f)             
            
            
        if datatype=='已处理数据' and datatype_save:
            pj_name=get_pj_name() 
            project_sel_dir=os.path.join(base_path, 'projects',pj_name) 
            project_sel_pj_configs_dir=os.path.join(project_sel_dir, 'pj_configs.json')
            
            with open(project_sel_pj_configs_dir,'r') as f:
                pj_configs=json.load(f)  
            pj_configs['data_source']='processed_data'              
            with open(project_sel_pj_configs_dir,'w') as f:
                json.dump(pj_configs,f)               
            
            
    pj_name=get_pj_name()  
    project_sel_dir=os.path.join(base_path, 'projects',pj_name) 
    project_sel_pj_configs_dir=os.path.join(project_sel_dir, 'pj_configs.json')      
    data_name=get_data_name(pj_name)
    data_source=get_data_source(pj_name) 
    st.write('当前数据源',data_source)
    st.markdown('')
    
    tab1,tab2,tab3 ,tab4 ,tab5,tab6,tab7=  st.tabs(["数据概览",
                                                    "参数录入",                                            
                                                    "频谱分析", 
                                                    "时频分析",
                                                    "特征提取",
                                                    "故障诊断",
                                                    "仿真测试"]) 
    pj_name=get_pj_name()  
    data_name=get_data_name(pj_name)
    dataframe=data_read_cache(data_name,pj_name,data_source)
    cols=dataframe.columns
    with open(os.path.join(project_sel_dir, 'device_configs.yaml') ) as f:              
        device_configs = yaml.load(f, Loader=yaml.SafeLoader)    

        
    with tab1: 
        col_sel=st.selectbox('选择列',cols)
        dataframe=data_read_cache(data_name,pj_name,data_source)[[col_sel]] #获取数据内容和数据文件名  
        "原始数据"
        with st.expander('数据内容'):
            dataframe
        "数据描述"
        st.table(dataframe.describe())
        
        "数据类型"
        for col in dataframe.columns:
            st.write(col,dataframe[col].dtypes)
            
            
    with tab2:        
        
        device_type=st.selectbox('滚动轴承',['滚动轴承','滑动轴承','齿轮','转子','其他'],key='device')  
        device_configs['device_type']=device_type
        
        
        def default_val(device_configs,para_name):
            if para_name in device_configs:
                return device_configs[para_name]
            else:
                return 1.00
        
        if device_type=='滚动轴承':
            if 'sensor_type' in device_configs:
                index_v=(['速度','位移','加速度','麦克风']).index(device_configs['sensor_type'])
            else:
                index_v=0
            sensor_type=st.selectbox('传感器类型',['速度','位移','加速度','麦克风'],index=index_v)  
          
            D=st.number_input('轴承滚道节径',
                                      value=default_val(device_configs,'D'))         
            d=st.number_input('滚珠直径',
                                      value=default_val(device_configs,'d'))
            n=st.number_input('滚珠数目',
                                      value=default_val(device_configs,'n'))
            fo=st.number_input('外圈频率',
                                      value=default_val(device_configs,'fo')) 
            
            fi=st.number_input('内圈频率',
                                      value=default_val(device_configs,'fi')) 
            
            device_configs['sensor_type']=sensor_type
            device_configs['D']=D
            device_configs['d']=d
            device_configs['n']=n
            device_configs['fo']=fo
            device_configs['fi']=fi            
            
            
#         if device_type=='齿轮':
#             sensor_type=st.selectbox('传感器类型',['速度','位移','加速度','麦克风'])            
#             big1=st.number_input('大齿轮齿数目')         
#             big1=st.number_input('大齿轮转速rpm')
#             small1=st.number_input('小齿轮齿数目')         
#             small2=st.number_input('小齿轮转速rpm')           
            
        data_freq=st.number_input('数采频率',value=25600) 
        device_configs['data_freq']=data_freq
        device_configs['RFI']=fi
               
        col1,col2=st.columns(2)
        with col1:
            if st.button('故障频率计算',key='故障频率'):
                fault_freq={}
                fault_freq['RFI']=fi
                fault_freq['BPFI']=np.float(np.abs(n*(fo-fi)*(1+d/D)/2))
                fault_freq['BPFO']=np.float(np.abs(n*(fo-fi)*(1-d/D)/2))
                fault_freq['FTF']=np.float(np.abs(fo*(1+d/D)/2+fi*(1-d/D)/2))
                fault_freq['BSF']=np.float(np.abs((fo-fi)*D/d*(1-(d*d/D/D))/2))
                

                # fault_freq={'RFI':RFI,'BPFI':round(BPFI)}
                fault_freq
                
                with open(os.path.join(project_sel_dir, 'fault_freq.yaml'),'w') as f:                             
                    yaml.dump(fault_freq,f, allow_unicode=True)  
                st.success('故障频率保存成功')
                
        with col2:
            if st.button('设备参数保存',key='参数保存'):
                with open(os.path.join(project_sel_dir, 'device_configs.yaml'),'w') as f:                             
                    yaml.dump(device_configs,f, allow_unicode=True)  
                    
                st.success('设备参数保存成功')
  
                

                    
    with tab3: 
        data1d=data_read_cache(data_name,pj_name,data_source)[[col_sel]].values.T[0] #获取数据内容和数据文件名                 
        txt = st.text_area('FFT','''1,最大分辨频率=采样频率/2；
2,频率分辨率=最大频率/样本数目；
3,样本数目最好是采样频率的整数倍，否则会造成频率泄露；
4,FFT只能针对稳态信号处理；''')
        
        st.write('解调(低频信号解析)')
        demodulation=st.checkbox('',key='demodulation0')
        sample_rate=st.number_input('采样频率(hz)',value=int(data_freq), min_value = 1, max_value = 1000000,key='sample_rate0')
        fft_plot_max_freq=st.slider('最大显示频率',min_value=0.0,max_value=sample_rate/2,value=sample_rate/2)
        
        ##频域图
        if os.path.exists(os.path.join(project_sel_dir, 'fault_freq.yaml')) and os.path.exists(os.path.join(project_sel_dir, 'device_configs.yaml')): 
            freq_amp=fft_plot(data1d,sample_rate,st,demodulation,fft_plot_max_freq=fft_plot_max_freq)        
            freq_amp
        else:
            st.warning('请先设定设备参数，计算故障频率')
        
        # ##时域图
        p = figure(
                    title='时域波形',
                    x_axis_label='time',
                    y_axis_label='Amplitude',
                    width=1000, 
                    height=400, 
                    background_fill_color="black")        
        p.line(range(len(data1d)),data1d)
        st.bokeh_chart(p)        
        
        
    with tab4:

        col1,col2,col3=st.columns(3)
        method=col1.selectbox('方法',['短时傅里叶','连续小波变换'])
        sample_rate=col2.number_input('采样频率(hz)',value=int(data_freq), min_value = 1, max_value = 1000000,key='sample_rate1')

        if method=='短时傅里叶':
            
            window_len=col3.number_input('窗口大小',value=1024, min_value = 1, max_value = 100000)      
            
            fig, ax = plt.subplots(figsize=(15,4))
            plt.specgram(data1d,NFFT=int(window_len),Fs=int(sample_rate),noverlap=int(window_len/2))

            plt.ylabel('频率',fontsize=20)
            plt.xlabel('时间',fontsize=20)
            st.pyplot(fig)
            
             
        
        if method=='连续小波变换':
            t = np.arange(0, len(data1d)/sample_rate, 1.0 / sample_rate)
            wavename = 'cgau8'
            totalscal = 256
            fc = pywt.central_frequency(wavename)
            cparam = 2 * fc * totalscal
            scales = cparam / np.arange(totalscal, 1, -1)
            [cwtmatr, frequencies] = pywt.cwt(data1d, scales, wavename, 1.0 / sample_rate)
            fig, ax = plt.subplots(figsize=(15,4))
       
            plt.contourf(t, frequencies, abs(cwtmatr))
            plt.ylabel(u"频率(Hz)")
            plt.xlabel(u"时间(秒)")
            plt.subplots_adjust(hspace=0.4)
            st.pyplot(fig)
            
        ##时域图
        p = figure(
                    title='时域波形',
                    x_axis_label='time',
                    y_axis_label='Amplitude',
                    width=700, 
                    height=200, 
                    background_fill_color="black")        
        p.line(range(len(data1d)),data1d)
        st.bokeh_chart(p)      
        
    with tab5:
     
        data1d=data_read_cache(data_name,pj_name,data_source)[[col_sel]].values.T[0] #获取数据内容和数据文件名     
        fea_ana=st.selectbox('特征分析', ['时域特征','频域特征'])
        if fea_ana=='频域特征' :
            hilbert_c=st.checkbox('包络解调')
            freqbase_error=st.number_input('频率偏差',key='频率偏差') 
        start_ana=st.button('开始计算')
        if fea_ana=='时域特征' and start_ana:
            time_fea_name=['均值','绝对平均值','方差','标准差','方根幅值','均方根值','峰值','最大值','最小值',
               '波形指标','峰值指标','脉冲指标','裕度指标','偏斜度','峭度']
            time_domain_features=pd.DataFrame(columns=time_fea_name)
            data1d=pd.DataFrame(data1d).T
            for i in range(len(data1d)):
                
                time_fea=pd.DataFrame(get_time_domain_features(data1d.iloc[i:i+1,:])).T
                time_fea.columns=time_fea_name
                time_domain_features=time_domain_features.append(time_fea)
            time_domain_features.index=range(len(data1d))
            time_domain_features

        if fea_ana=='频域特征' :

            freq_amp=get_freq_domain_features(data1d,hilbert_c,sample_rate)
            try:
                with open(os.path.join(project_sel_dir, 'fault_freq.yaml')) as f:                             
                    fault_freq=yaml.load(f, Loader=yaml.SafeLoader)
            except:
                st.warning('请先计算故障频率')

            fault_freq
            st.write('特征结果：')
            try:
                for key in fault_freq:
                    for i in range(1,4):
                        fault_freq_low=(fault_freq[key]-freqbase_error)*i
                        fault_freq_high=(fault_freq[key]+freqbase_error)*i
                        fault_real_amp_scale=freq_amp.loc[fault_freq_low:fault_freq_high]
                        # fault_real_amp_scale
                        fault_real_amp=fault_real_amp_scale.max()[0]
                        fault_real_freq=fault_real_amp_scale.idxmax()[0]
      
                        
                        st.write(key+'乘'+str(i)+'-----','理论故障频率：',round(fault_freq[key]*i,2),
                                 '实际故障频率：',round(fault_real_freq,2),
                                 '故障频率amp：',round(fault_real_amp,2))
            except:
                st.warning('频率偏差过低')


       
    with tab6:                
        
        with open(os.path.join(project_sel_dir, 'device_configs.yaml')) as f:                  
            device_configs = yaml.load(f, Loader=yaml.SafeLoader)  
            
            
        if device_configs['device_type']=='滚动轴承':
            
            if os.path.exists(os.path.join(project_sel_dir, 'fault_freq.yaml')): 
                with open(os.path.join(project_sel_dir, 'fault_freq.yaml')) as f:                             
                    fault_freq=yaml.load(f, Loader=yaml.SafeLoader) 
                    
                    st.subheader("1，故障定义")
                    
                    col1,col2=st.columns(2)  
                    diag_name=col1.text_input('诊断案例命名')
                    col1,col2=st.columns(2)
                    freqbase=col1.selectbox('故障频率',fault_freq)
                    col2.write('故障频率(hz)')
                    col2.write(fault_freq[freqbase])  
                    
                    col1,col2=st.columns(2)
                    freqbase_error=col1.number_input('频率偏差') 
                    with col2:
                        st.write('故障频率范围(hz)')
                        
                        st.write(fault_freq[freqbase]-freqbase_error,fault_freq[freqbase]+freqbase_error)                  
                    
                    col1,col2=st.columns(2)
                    n_freqbase=col1.number_input('倍频系数', min_value=1,max_value=10)
                    with col2:
                        st.write('故障频率倍频范围(hz)')
                        freq_fault_low=n_freqbase*(fault_freq[freqbase]-freqbase_error)
                        freq_fault_high=n_freqbase*(fault_freq[freqbase]+freqbase_error)                      
                        st.write(freq_fault_low,freq_fault_high)   
                        
                    col1,col2=st.columns(2)    
                    limit=col1.number_input('报警限值')
    
                    diag_para={'device_type':device_configs['device_type'],
                                'diag_name':diag_name,
                                'freq_fault_low':freq_fault_low,
                                'freq_fault_high':freq_fault_high
                                }
                    
       
     
                    if os.path.exists(os.path.join(project_sel_dir, 'diag_paras.yaml')):
                        with open(os.path.join(project_sel_dir, 'diag_paras.yaml')) as f:
                              
                            diag_paras = yaml.load(f, Loader=yaml.SafeLoader) 
                            diag_paras[diag_name]=diag_para
                    else:
                        diag_paras={diag_name:diag_para}
                    
                    save_diag_para=st.button('诊断参数保存')
                    if save_diag_para:
                        with open(os.path.join(project_sel_dir, 'diag_paras.yaml'),'w') as f:
                            yaml.dump(diag_paras,f, allow_unicode=True) 
                        st.success('诊断已保存')
                        
                    st.subheader('2，故障诊断')
                    
                    if os.path.exists(os.path.join(project_sel_dir, 'diag_paras.yaml')):
                        with open(os.path.join(project_sel_dir, 'diag_paras.yaml')) as f:
                            diag_all = yaml.load(f, Loader=yaml.SafeLoader) 
                        
                        diags=st.multiselect('选择诊断名称', diag_all)                
                        with st.expander('所有故障定义信息'):
                            diag_all
                            
                        col1,col2,col3=st.columns(3)
                        demodulation=col1.checkbox('解调',key='解调')
                        startdiag=col2.checkbox('开始诊断')
                        deldiag=col3.button('删除所选故障')
                        
                        if deldiag  and len(diags):
                            for key in diags:
                                del diag_all[key]
                            with open(os.path.join(project_sel_dir, 'diag_paras.yaml'),'w') as f:
                                yaml.dump(diag_all,f, allow_unicode=True) 
                            
                        if startdiag  and len(diags):
                            data1d=data_read_cache(data_name,pj_name,data_source)[[col_sel]].values.T[0] #获取数据内容和数据文件                    
                            ##频域图
                            fft_plot_max_freq=st.slider('最大显示频率',min_value=0.0,max_value=sample_rate/2,value=sample_rate/2,key='freqmax')
                            freq_amp=fft_plot(data1d,data_freq,st,demodulation,diag_det=True,diag_para=diag_all,diag_sel=diags,fft_plot_max_freq=fft_plot_max_freq)
                            ##时域图
                            p = figure(
                                        title='时域波形',
                                        x_axis_label='time',
                                        y_axis_label='Amplitude',
                                        width=1000, 
                                        height=400, 
                                        background_fill_color="black")        
                            p.line(range(len(data1d)),data1d)                       
                            st.bokeh_chart(p)  
                    else:
                        st.info('请先进行故障定义')
            else:
                st.warning('请先计算故障频率')
                    
                                
                
                               
    with tab7:
        col1,col2=st.columns(2)
        with col1:
            sample_rate=st.number_input('采样频率(hz)',value=1000, min_value = 1, max_value = 1000000)
        with col2:
            data_len=st.number_input('样本长度', value=1000,min_value = 1, max_value = 1000000)
        t=signalt_generate(sample_rate,data_len)
        
        col1,col2=st.columns([5,1])
        with col1:
            signal_define=st.text_input('定义信号',value='0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)')
        st.caption('调幅调频信号定义(from scipy samples,采样频率=400)：')
        st.caption('chirp(np.arange(400)/400,20.0, (np.arange(400)/400)[-1],100.0) * (1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * (np.arange(400)/400)))')
        
        with col2:
            st.write('解调(低频信号解析)')
            demodulation=st.checkbox('')
        X =eval(signal_define)

        freq_amp=fft_plot(X,sample_rate,st,demodulation) 
        freq_amp
        p = figure(
        title='时域信号(如果解调，则黄色线为包络谱图)',
        x_axis_label='时间(S)',
        y_axis_label='Amp',
        width=1000, height=400, background_fill_color="black")
        p.line(range(len(X)),X,color='white')
        if demodulation:
            p.line(range(len(X)),np.abs(hilbert(X)).tolist(),color='yellow',line_width=5)
        p.circle(range(len(X)),X,color='red')
        st.bokeh_chart(p)          
        instantaneous_phase = np.unwrap(np.angle(hilbert(X)))
        instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * sample_rate) 
        