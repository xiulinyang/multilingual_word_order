from conllu import parse
from pathlib import Path
import glob
from tqdm import tqdm
sv = 0
vs = 0
ov = 0
vo = 0

with open('en_parallel_ov_unfill.txt', 'w') as parallel, open('fi_different_vo_unfill.txt', 'w') as different:
    for path in glob.glob('english/*.conllu'):
        ewt_test = Path(path).read_text().strip().split('\n\n')
        for ewt in tqdm(ewt_test):
            ewt_parsed = parse(ewt)[0]
            root = [x['id'] for x in ewt_parsed if x['deprel']=='root'][0]
            subj_id = [x['id'] for x in ewt_parsed if x['deprel']=='nsubj' and x['head']==root]
            obj_id = [x['id'] for x in ewt_parsed if x['deprel'] in ['obj'] and x['head']==root]
            if obj_id:
                # subj_text = [x['form'] for x in ewt_parsed if type(x['id'])==int and x['id'] <= subj_id[0]]
                root_text = [x['form'] for x in ewt_parsed if type(x['id'])==int and x['id'] < root]
                rest_root_text = [x['form'] for x in ewt_parsed if type(x['id'])==int and x['id'] > root]
                root_token = [x['form'] for x in ewt_parsed if type(x['id'])==int and x['id'] == root][0]
                obj_text = [x['form'] for x in ewt_parsed if type(x['id']) == int and x['id'] < obj_id[0]]
                rest_obj_text = [x['form'] for x in ewt_parsed if type(x['id']) == int and x['id'] > obj_id[0]]
                obj_token = [x['form'] for x in ewt_parsed if type(x['id']) == int and x['id'] == obj_id[0]][0]

                if obj_id[0] < root:
                    # if len(subj_text)>2:
                    obj_text = ' '.join(obj_text + ['[MASK]'] + rest_obj_text)+'\t'+ obj_token
                    parallel.write(f'{obj_text}\n')
                    ov += 1
                elif root < obj_id[0]:
                    # if len(root_text)>2:
                    root_text = ' '.join(root_text+['[MASK]'] + rest_root_text) + '\t' + root_token
                    different.write(f'{root_text}\n')
                    vo += 1

# print(f'svo: {svo}\nsov: {sov}\nvso: {vso}\nvos: {vos}')
print(f'vo:{vo}, ov:{ov}')