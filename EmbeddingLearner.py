import nltk
from gensim.models.word2vec import Word2Vec
import string
from string import digits
import scp
import paramiko
from scp import SCPClient, SCPException
import os
import re
from tqdm import trange
import multiprocessing
from multiprocessing import Process, Pool
from gensim.models import KeyedVectors
import yaml

cli=None
translator=None

def pre_process(text):
    #https://blog.naver.com/PostView.nhn?blogId=timtaeil&logNo=221361106051&redirect=Dlog&widgetTypeCall=true&directAccess=false
    #Delete former words of ':' Because it contain dates and host name.
    #Example : Mar 23 06:37:33 225-2c-1 10freedos: debug: /dev/vda15 is a FAT32 partition
    #Remove number, symbol, stop world
    #Tokenizing
    #clean = [[x.lower() for x in each[each.find(':',19)+1:].translate(translator).split() \
    #       if x.lower() not in stop_words] for each in text.split('\n')]
    if text=='':
        return []
    global translator
    clean = [[x.lower() for x in each[each.find(':',19)+1:].translate(translator
        ).split() ] for each in text.split('\n')]
    return clean

def download_log(remote_path, file_name, local_path):
    global cli
    try:
        with SCPClient(cli.get_transport()) as scp:
            scp.get(remote_path+file_name, local_path)
    except SCPException as e:
        print("Operation error : %s"%e)
    try:
        with open(local_path+file_name) as f:
            text = f.read()
    except:
        try:
            with open(local_path+file_name, encoding='ISO-8859-1') as f:
                text = f.read()
        except Exception as e:
            print("Opreation error at reading file %s : %s"%(file_name, e))

    try :
        text = re.sub(r'[0-9]+', '', text)
    except Exception as e:
        print("Operation error at re.sub : %s"%e)
        os.remove(local_path+file_name)
        return ''
    os.remove(local_path+file_name)
    return text

def get_file_list(path):
    global cli
    stdin, stdout, stderr = cli.exec_command('ls '+path)
    rslt=stdout.read()
    file_list=rslt.split()
    del stdin, stdout, stderr
    file_list = [file_name.decode('utf-8') for file_name in file_list]
    return file_list

def multi_reading(tmp):
    path, file_name, local_path = tmp
    log = download_log(path, file_name, local_path)
    #Pre processing
    log_token=pre_process(log)
    return log_token


if __name__ == '__main__':
    local_path='/home/dpnm/tmp/'
    remote_path='/mnt/hdd/log/'#225-2c-4/04-23/'
    #file_name='snort.log'
    with open ('../server_info.yaml') as f:
        server_info=yaml.load(f)['log']
    #nltk.download('stopwords')
    #stop_words=set(nltk.corpus.stopwords.words('english'))
    #print (stop_words)
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    cli=paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    cli.connect(server_info['ip'], port=22, username=server_info['id'], password=server_info['pwd'])
    model_= Word2Vec.load('embedding_with_log')


    log_corpus=[]
    #Get file names and paths of all log file except sudo.log, CRON.log, stress-ng.log
    host_list = get_file_list(remote_path)
    for i in range(len(host_list)):
        if i<4:continue
        host=host_list[i]
        print( "Reading from %d th HOST start. Total : %d"%(i+1, len(host_list)) )
        print(" -------HOST name : "+host+"-------------")
        log_dir_list=get_file_list(remote_path+host)
        for j in trange(len(log_dir_list)):
            log_dir=log_dir_list[j]
            #if j%2==1: #use only half of data. Too many.
            #    continue
            log_file_list=get_file_list(remote_path+host+'/'+log_dir)
            path_=remote_path+host+'/'+log_dir+'/'
            #Download log file from Monitoring node
            #Tryed to change to use multi processing, but failed ( became slower )
            '''
            tmp_=[[path_, file_name, local_path] for file_name in log_file_list]
            with Pool() as p:
                log_parsing_result=p.map(multi_reading,tmp_)
            for log_token in log_parsing_result:
                log_corpus.extend(log_token)
            '''
            for file_name in log_file_list:
                if file_name in ['sudo.log', 'CRON.log', 'stress-ng.log', 'apache-access.log']:
                    continue
                log = download_log(path_,file_name,local_path)
                log_token=pre_process(log)
                log_corpus.extend(log_token)
            #print ("Readed %d sentences of log"%len(log_corpus))
            if not 'model_' in locals():
                model_=Word2Vec(log_corpus, min_count=2, window=5, sg=1, vector_size=100, epochs=1, workers=multiprocessing.cpu_count())
            else:
                model_.build_vocab(log_corpus, update=True)
                model_.train(log_corpus, total_examples=len(log_corpus), epochs=5)
            log_corpus=[]
        model_.save('embedding_with_log')
        print('model saved. it contain %d number of words'%len(model_.wv))


    #Word2Vec Learning
    print('start Word2Vec learning')
    '''min_list=[2,5]
    window_list=[3,5]
    size_list=[100,300,500]
    for min_ in min_list:
        for window_ in window_list:
            for size_ in size_list:
                model_=Word2Vec(log_corpus,min_count=min_, window=window_,size=size_, sg=1, iter=10000, workers=4)
                model_.wv.save_word2vec_format('embedding_min%d_win_%d_size_%d'%(min_,window_,size_))
                print('Embedding with min%d_win_%d_size_%d Saved!'%(min_,window_,size_))
                '''

