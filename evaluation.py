import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import gc
import re
import json
import torch
import math
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from tqdm import tqdm
import numpy as np 
import pandas as pd
from glob import glob
import random
from scipy.stats import pearsonr
import argparse

from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel

chars=['A','B','C','D','E','F','G','H','I','J','K','L','M']
def check_span_overlap(span1,span2,by_overlap=False):
    span1=span1.replace('{','').replace('','}')
    span2=span2.replace('{','').replace('','}')
    if span1 and span2:
        if by_overlap:
            if span1 in span2 or span2 in span1:
                return True
            min_len = min(len(span1), len(span2))
            
            for i in range(2, min_len + 1):
                if span1[-i:] == span2[:i] or span2[-i:] == span1[:i]:
                    return True
        else:
            return span1.strip()==span2.strip()
    return False

def count_span_overlap(span1,span2):
    score=0
    if span1 and span2:
        min_len=min(len(span1),len(span2))
        if span1 in span2 or span2 in span1:
            return min_len
        for i in range(1, min_len + 1):
            if span1[-i:] == span2[:i] or span2[-i:] == span1[:i]:
                score=i
    return score


def read_events(span,temp_trigger=None):
    result=[]
    if ':' in span:
        temp_trigger=span[:span.index(':')].strip().replace(' - ',',').replace('-',',')
        span=span[span.index(':')+1:].strip()
        if span.strip()!='None':
            result=[s.strip() for s in span.split('...') if s.strip()]
            if len(result)>20:
                result=list(set(result))
    return result,temp_trigger

def calculate_event_scores(scores,reference,prediction,trigger,FN,FP):
    reference,trigger=read_events(reference)
    prediction,pred_trigger=read_events(prediction)
    # if pred_trigger!=trigger:
    #     prediction=[]
    #print(reference,prediction)
    if ',' not in trigger:
        key=f'{trigger},trigger,span'
    else:
        key=trigger
    key=key.replace(' - ',',').replace('-',',')
    if key.count(',')>2:
        key[key.index(',')]='_'
    
    scores[key]=scores.get(key,[0,0,0])
    scores[key][0]+=len(reference)
    scores[key][1]+=len(prediction)

    
    for p in prediction:
        ptrue=False
        for idx,r in enumerate(reference):
            if r and check_span_overlap(p.lower(),r.lower()):
                scores[key][2]+=1
                reference[idx]=None
                ptrue=True
                break
        if not ptrue:
            FP.append((trigger,p))
    for r in reference:
        if r is not None:
            FN.append((trigger,r))
    return scores,FN,FP

def calculate_PICO_scores(scores,reference,prediction,trigger,FN,FP):
    reference,trigger=read_events(reference)
    prediction,pred_trigger=read_events(prediction)
    # if pred_trigger!=trigger:
    #     prediction=[]
    #print(reference,prediction)
    if ',' not in trigger:
        key=f'{trigger},trigger,span'
    else:
        key=trigger
    key=key.replace(' - ',',').replace('-',',')
    if key.count(',')>2:
        key[key.index(',')]='_'
    
    reference=[ w.strip() for r in reference for w in r.split() if w.strip()]
    prediction=[w.strip() for p in prediction for w in p.split() if w.strip()]

    scores[key]=scores.get(key,[0,0,0])
    scores[key][0]+=len(reference)
    scores[key][1]+=len(prediction)
    for p in prediction:
        find_match=False
        for idx,r in enumerate(reference):
            if r and r==p:
                scores[key][2]+=1
                reference[idx]=None
                find_match=True
                break
        if not find_match:
            FP.append((trigger,p))
    for r in reference:
        if r is not None:
            FN.append((trigger,r))
    return scores,FN,FP

def write_scores(scores,filename,doc_id=None):
    exclude_rows=[',no,',',not mentioned,','not related to','cannot be','can not be','none,']
    overall=[0,0,0]
    for k,v in scores.items():
        if not any([r in k.lower()+',' for r in exclude_rows]) and v[0]>0:
            for i in range(3):
                overall[i]+=v[i]

    filedir='/'.join(filename.split('/')[:-1])
    os.makedirs(filedir, exist_ok=True)
    scores['Overall,-,-']=overall
    with open(filename,'w') as f:
        f.write('triiger,attribute,subtype,NT,NP,TP,P,R,F1\n')
        for k,[NT,NP,TP] in scores.items():
            if NT==0 or any([r in k.lower()+',' for r in exclude_rows]):
                continue
            P=TP*100/NP if NP else 0
            R=TP*100/NT if NT else 0
            F1=2*P*R/(P+R) if P+R else 0

            if ',' not in k:
                k='relation,'+k+','
            if doc_id:
                f.write(f'{doc_id},{k},{NT},{NP},{TP},{P},{R},{F1}\n')
            else:
                f.write(f'{k},{NT},{NP},{TP},{P},{R},{F1}\n')

    return

def evaluate_events(dic,scores,FN,FP,key=None,task='events'):

    if key is None or key in dic['prompt_id']:
        if '\n' in dic['reference']:
            references=[l.strip() for l in dic['reference'].split('\n') if l.strip()]
            predictions=[l.strip() for l in dic['prediction'].split('\n') if l.strip()]

            #geting invalid scores in this case
            if len(references)!=len(predictions):
                copy_predictions=copy.deepcopy(predictions)
                predictions=['None' for c in references]
                for cp in copy_predictions:
                    if ':' in cp:
                        entity_type=cp.split(':')[0]
                        for idx,r in enumerate(references):
                            if r[:len(entity_type)]==entity_type:
                                predictions[idx]=cp
                                break                
            for ir,ip in zip(references,predictions):
                if task=='PICO':
                    scores,FN,FP=calculate_PICO_scores(scores,ir,ip,dic['type'],FN,FP)
                else:
                    scores,FN,FP=calculate_event_scores(scores,ir,ip,dic['type'],FN,FP)
        else:
            scores,FN,FP=calculate_event_scores(scores,dic['reference'],dic['prediction'],dic['type'],FN,FP)
    
    return scores,FN,FP

def read_options(options,output):
    answer=options[0].replace(',','-').replace('associated_with_someone_else','not_patient')
    for i,op in enumerate(options):
        if f"({chars[i]})" in output:
            answer=op.replace(',','-').replace('associated_with_someone_else','not_patient')
            break
    return answer

def evaluate_classification(dic,scores={}):
    FP,FN=[],[]
    options=dic['options'].split('[SEP]')
    #print(options)
    ref,pred=read_options(options,dic['reference']),read_options(options,dic['prediction'])

    # if pred and 'int' in pred:
    #     pred=None
    type=dic["type"].replace(' - ',',').replace('-',',')
    while type.count(',')>1:
        type=list(type)
        type[type.index(',')]='_'
        type=''.join(type)
    if ref:
        ref=f'{type},{ref}'
        if ref.count(',')==1:
            ref+=',-'
        scores[ref]=scores.get(ref,[0,0,0])
        scores[ref][0]+=1
    if pred:
        pred=f'{type},{pred}'
        if pred.count(',')==1:
            pred+=',-'
        scores[pred]=scores.get(pred,[0,0,0])
        scores[pred][1]+=1            
        if pred==ref:
            scores[ref][2]+=1
        else:
            FP=pred
            FN=ref
    return scores,FN,FP

def evaluate_MultiClassification(dic,scores={}):
    FP,FN=[],[]
    options=dic['options'].split('[SEP]')
    #print(options)
    ref=[op for i,op in enumerate(options) if f'({chars[i]})' in dic['reference']]
    pred=[op for i,op in enumerate(options) if f'({chars[i]})' in dic['prediction']]

    for op in ref:
        scores[op]=scores.get(op,[0,0,0])
        scores[op][0]+=1
        if op in pred:
            scores[op][2]+=1
    
    for op in pred:
        scores[op]=scores.get(op,[0,0,0])
        scores[op][1]+=1    

    if pred!=ref:
        FP=pred
        FN=ref
    return scores,FN,FP

def evaluate_accuracy(dic,scores={}):
    FP,FN=[],[]

    stype='classification,,'
    scores[stype]=scores.get(stype,[0,0,0])
    scores[stype][0]+=1
    scores[stype][1]+=1
    answer_wrong=True
    if dic['reference'].strip().lower() in dic['prediction'].strip().lower():
        scores[stype][2]+=1
        answer_wrong=False
    # elif not any([f'({c})' in dic['prediction'] for c in chars]):
    #     #print(options)
    #     options=dic["options"].split('[SEP]')
    #     for idx,op in enumerate(options):
    #         if op=='no':
    #             options[idx]='not'
    #         elif op=='yes':
    #             options[idx]=='is'
    #         if op in dic['reference'] and op in dic['prediction']:
    #             answer_wrong=False
    #             break

    if answer_wrong:
        FP=dic['prediction'][3:].strip()
        FN=dic['reference'][3:].strip()
    
    return scores,FP,FN

def summarize_scores(exp):
    source_files =glob(f'predictions/NLU/{exp}/**/*.csv')
    #print(f'predictions/NLU/{exp}/**/*.csv')
    # 2. Convert list of files into list of dataframes
    dataframes = []
    for filename in source_files:
        print(filename)
        df = pd.read_csv(filename) 
        task=filename.split('/')[-1].split('.')[0]
        category=filename.split('/')[-2]
        df.insert(0,'task',task)
        df.insert(0,'category',category)
        dataframes.append(df)

    # 3. Concatenate it
    df = pd.concat(dataframes)
    df.to_csv(f'predictions/NLU/{exp}_summary.csv',index=False)
    return

def evaluate_pearson_correlation(dics,error_file,score_file):
    pred,ref=[],[]
    for dic in dics:
        options=dic['options'].split('[SEP]')
        r,p=read_options(options,dic['reference']),read_options(options,dic['prediction'])
        pred.append(int(p))
        ref.append(int(r))
        if p!=r:
            with open(error_file,'a') as f:
                json.dump(dic,f)
                f.write('\n')
    with open(score_file.replace('.csv','.txt'),'w') as f:
        f.write(str(pearsonr(ref, pred)[0]*100))
    return
    
def evaluate_task(reference_file):
    category=reference_file.split('/')[-2]
    task=reference_file.split('/')[-3]

    predfile=f'predictions/NLU/{exp}_{N_shot}shot/{task}/{category}.jsonl'
    score_file=f'predictions/NLU/{exp}_{N_shot}shot/{task}/{category}.csv'

    error_file=predfile.replace('.jsonl','_error.jsonl')
    if not os.path.isfile(predfile):
        print(predfile)
        return
    with open(error_file,'w') as f:
        a=1
    scores={}
    
    dics=[json.loads(l) for l in open(predfile).read().split('\n') if l.strip()]
    for dic in dics:
        FN,FP=[],[]
        if dic['task'] in ['events','PICO']:
            scores,FN,FP=evaluate_events(dic,scores,FN,FP,task=dic['task'])
        elif dic['task']  in ['QA','NLI']:
            scores,FN,FP=evaluate_accuracy(dic,scores)
        elif dic['task'] in ['MultiQA']:
            scores,FN,FP=evaluate_MultiClassification(dic,scores)
        elif dic['task']!='STS':
            scores,FN,FP=evaluate_classification(dic,scores)
        else:
            continue
        if FP!=[] or FN!=[]:
            dic['FP']=FP
            dic['FN']=FN
            with open(error_file,'a') as f:
                json.dump(dic,f)
                f.write('\n')
    if dics and dics[0]['task']=='STS':
        evaluate_pearson_correlation(dics,error_file,score_file)


    write_scores(scores,score_file)
    return

def sample_messages(sample_dics,N_shot,target_options):
    result=[]
    answers,num_Nones=[],1
    count=0
    while len(sample_dics)>0 and count<N_shot:
        msa=random.sample(sample_dics,1)[0]
        answer=msa["messages"][-1]['content']
        ## \ msa['options']==target_options and
            
        if  answer not in answers and \
            (msa['task']!='events' or answer.count(': None')!=num_Nones):
            num_Nones=answer.count(': None')
            answers.append(answer)
            result.extend(msa["messages"][1:])
            count+=1
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--base_id', action="store")
    parser.add_argument('--outname', action="store")
    parser.add_argument('--lora', action="store",default='None')
    parser.add_argument('--N_shot', action="store",default=0)
    parser.add_argument('--task_category', action="store",default='**')
    args = parser.parse_args()

    # # Delete the llm object and free the memory
    # destroy_model_parallel()
    # #del llm
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group()

    # tasks that you don't want to include for evaluation
    tasks_to_exclude=[]
    num_samples=100000000
    N_shot=int(args.N_shot)#2


    #task_category= '**'#'classification' #'**' #for all task #'2023PedSHAC' #
    interested_task='**'
    
    task_category= args.task_category#'events' # #'attributes'#'classification' #'**' #for all task #'2023PedSHAC' #
    #interested_task='HoCQA'   

    task_dir=f'dataset/final/test/{task_category}'

    for base_model_id,ori_exp,lora_dir in [[args.base_id,args.outname,args.lora]]:
        exp=ori_exp#.split('/')[-]     
        torch.random.manual_seed(0)
        #.replace('/','-')
        if num_samples>0:
            
            ### loading the model
            llm = LLM(model=base_model_id, tokenizer=base_model_id, #enable_lora=True,
                    tensor_parallel_size=4
                    )
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            
            sampling_params = SamplingParams(temperature=0, 
                                            #top_k=-1,
                                            # logprobs=1,
                                            # prompt_logprobs=1,
                                            max_tokens=2048,
                                            #max_seq_len=2048,
                                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                                            #stop_token_ids=[tokenizer.eos_token_id]
                                            )

            # chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            # tokenizer.chat_template = chat_template
            
            for reference_file in glob(f'{task_dir}/{interested_task}/test.json'):
                if any([t in reference_file for t in tasks_to_exclude]): # or 'PICO' in reference_file
                    continue
                # if 'CACER' in reference_file or 'PICO' in reference_file: #'events' in reference_file or 
                #     continue            
                category=reference_file.split('/')[-2]
                task=reference_file.split('/')[-3]
                # getting the filenames
                os.makedirs(f'predictions/NLU/{exp}_{N_shot}shot/{task}',exist_ok=True)
                print(f'working on {task}/{category}')
                
                predfile=f'predictions/NLU/{exp}_{N_shot}shot/{task}/{category}.jsonl'
                score_file=f'predictions/NLU/{exp}_{N_shot}shot/{task}/{category}.csv'
                
                if not os.path.isfile(predfile):
                    with open(predfile,'w') as f:
                        a=1
                
                if N_shot>0:
                    sample_dics=json.loads(open(reference_file.replace('test.json','train.json')).read())

                dics=json.loads(open(reference_file).read())
                num_pred=len([l for l in open(predfile).read().split('\n') if l.strip()])
                
                prompts=[]
                dic_index=[]
                for i in range(num_pred,min(num_pred+num_samples,len(dics))):
                    if dics[i]['task']=="attributes": #"attributes"
                        continue
                    if N_shot>0:
                        message=[dics[i]["messages"][0]]+ \
                                sample_messages(sample_dics,N_shot,dics[i]['options'])+ \
                                [dics[i]["messages"][1]]
                    else:
                        message=dics[i]["messages"][:-1]
                    
                    #message[-1]['content']='Think step by step and answer the following question.\n'+message[-1]['content']
                    if 'llama' not in base_model_id.lower():
                        #message[0]['content']+'\n'+
                        # message[1]['content']="Based on your medical knowledge and the medical text below, answer the following question. Your answer should start with 'This medical text describes ...' and end with 'Therefore, the answer is ... \n"+message[1]['content']
                        #message[1]['content']='Answer the question below faithfully according to the medical text.\n'+message[1]['content']
                        #+"\nLet's Think step by step. Your answer should start with 'This medical text describes ...' and end with 'Therefore, the answer is ... '.\n"+message[1]['content']
                        message=message[1:]
                    if task=='events' and N_shot>0:
                        for idx in range(0,N_shot+1):
                            m_idx=idx*2
                            if 'llama' in base_model_id.lower():
                                m_idx+=1
                            instruct,text=message[m_idx]['content'].split('\nMedical text:')
                            answers=dics[i]["messages"][-1]['content'].split('\n')
                            answers='\n'.join([a.split(':')[0]+': {span}' for a in answers])
                        
                            message[m_idx]['content']=f"{instruct}\nYour answer should use the following format, with one entity type per line. The span refers to the original text span from the Medical text. Output None if there is no such span. Use `...' to separate multiple spans.\n{answers}\nMedical text:{text}"
                    #print(message)
                    prompt = tokenizer.apply_chat_template(message, 
                                                        #add_generation_prompt=True, 
                                                        tokenize=False)
                    prompts.append(prompt)
                    dic_index.append(i)
                if os.path.isdir(lora_dir):
                    outputs = llm.generate(prompts, 
                                        sampling_params, 
                                        lora_request=LoRARequest(lora_dir,1,lora_dir),
                                        )
                else:
                    outputs = llm.generate(prompts, 
                                        sampling_params, 
                                        )

                for oidx, output in enumerate(outputs):
                    i=oidx+num_pred
                    generated_text = output.outputs[0].text.replace('<|assistant|>\n','').replace('<|start_header_id|>assistant<|end_header_id|>','').replace('<|start_header_id|>','').replace('<|end_header_id|>','').replace('assistant','').strip().replace('[SEP]','...')
                    result={
                        'prompt_id':dics[dic_index[i]]['prompt_id'],
                        'reference':dics[dic_index[i]]['messages'][-1]['content'],
                        'prediction':generated_text,
                        'task':dics[dic_index[i]]['task'],
                        'type':dics[dic_index[i]]['type'],
                        'options':dics[dic_index[i]]['options']
                    }
                    with open(predfile,'a') as f:
                        json.dump(result,f)
                        f.write('\n')
                evaluate_task(reference_file)

            # # Delete the llm object and free the memory
            # destroy_model_parallel()
            # del llm.llm_engine.model_executor.driver_worker
            # del llm # Isn't necessary for releasing memory, but why not
            # gc.collect()
            # torch.cuda.empty_cache()

        for reference_file in glob(f'{task_dir}/{interested_task}/test.json'):
            if any([t in reference_file for t in tasks_to_exclude]): # or 'PICO' in reference_file
                    continue
            evaluate_task(reference_file)
        #print(reference_file)
    ### scores
    summarize_scores(ori_exp)
