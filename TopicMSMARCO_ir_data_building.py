# encoding=utf-8
from tqdm import tqdm
import os
import pickle
import random
random.seed(0)

collection_path = '...path_to_your_dir/Documents/MSMARCO_data/collection.tsv'
query_path = "...path_to_your_dir/Documents/MSMARCO_data/queries.train.tsv"

collection_tokenized_path = '...path_to_your_dir/Documents/MSMARCO_data/collection.tokenized.pkl'
query_tokenized_path = '...path_to_your_dir/Documents/MSMARCO_data/queries.train.tokenized.pkl'

query_bert_path = '...path_to_your_dir/Documents/MSMARCO_data/queries.train.bert.pkl'
collection_bert_path = '...path_to_your_dir/Documents/MSMARCO_data/collection.bert.pkl'

# for the local features of DUET
tf_idf = '/home/lunet/cojh6/Documents/MSMARCO_data/idf.norm.token.pkl'
train_path = '...path_to_your_dir/MSMARCO_data/triples.train.topic_{}.tsv'
test_path = '...path_to_your_dir/MSMARCO_data/triples.test.topic_{}.tsv'

out_path = '...path_to_your_dataset_dir/'

topics = ['IT', 'furnishing', 'food', 'health', 'tourism', 'finance']


def load_queries(query_path):
    with open(query_path, 'rb') as f:
        queries = pickle.load(f)
    return queries


def load_collections(collection_path):
    with open(collection_path, 'rb') as f:
        docs = pickle.load(f)
    return docs


def process_train_triples(topic):
    path = train_path.format(topic)
    q_docs, pos_docs, neg_docs = [], [], []
    if os.path.exists(path):
        total = 0
        with open(path, 'r', encoding='utf-8') as fe:
            lines = fe.readlines()[:]
            pbar = tqdm(total=len(lines), desc='reading: ' + topic)
            for i, s_line in enumerate(lines):
                line = s_line.strip().split('\t')
                _qid, _pid, _nid = line[0], line[1], line[2]
                _q = queries[_qid]
                _pdoc = doc_collections[_pid]
                _ndoc = doc_collections[_nid]
                q_docs.append(_q)
                pos_docs.append(_pdoc)
                # pos_docs.append(_q*2)
                neg_docs.append(_ndoc)
                pbar.update(1)
                total += 1

    with open(os.path.join(out_path, 'topic.demo.{}.train.queries.pkl'.format(topic)), 'wb') as f:
        pickle.dump(q_docs, f)
    print(f"query tokenizeds saved!")
    with open(os.path.join(out_path, 'topic.demo.{}.train.pos.pkl'.format(topic)), 'wb') as f:
        pickle.dump(pos_docs, f)
    print(f"pos tokenizeds saved!")
    with open(os.path.join(out_path, 'topic.demo.{}.train.neg.pkl'.format(topic)), 'wb') as f:
        pickle.dump(neg_docs, f)
    print(f"neg tokenizeds saved!")


def process_test_triples(topic):
    path = test_path.format(topic)
    q_docs, pos_docs, labels = [], [], []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fe:
            lines = fe.readlines()
            pbar = tqdm(total=len(lines), desc='reading: ' + topic)
            used_pid = []
            for i, s_line in enumerate(lines):
                line = s_line.strip().split('\t')
                _qid, _pid, _nid = line[0], line[1], line[2]
                _q = queries[_qid]
                _ndoc = doc_collections[_nid]
                if (_qid, _pid) not in used_pid:
                    _pdoc = doc_collections[_pid]
                    q_docs.append(_q)
                    pos_docs.append(_pdoc)
                    # pos_docs.append(_q*2)
                    labels.append(1)
                    used_pid.append((_qid, _pid))
                q_docs.append(_q)
                pos_docs.append(_ndoc)
                labels.append(0)
                pbar.update(1)

    idx = list(range(len(labels)))
    random.shuffle(idx)
    labels = [labels[i] for i in idx]
    pos_docs = [pos_docs[i] for i in idx]
    q_docs = [q_docs[i] for i in idx]

    with open(os.path.join(out_path, 'topic.{}.test.label.pkl'.format(topic)), 'wb') as f:
        pickle.dump(labels, f)
    print(f"neg tokenizeds saved!")
    with open(os.path.join(out_path, 'topic.{}.test.doc.pkl'.format(topic)), 'wb') as f:
        pickle.dump(pos_docs, f)
    print(f"pos tokenizeds saved!")
    with open(os.path.join(out_path, 'topic.{}.test.queries.pkl'.format(topic)), 'wb') as f:
        pickle.dump(q_docs, f)


def process_duet_local_features():

    _files = ['topic.{}.train.queries.pkl', 'topic.{}.train.pos.pkl', 'topic.{}.train.neg.pkl',
              'topic.{}.test.queries.pkl', 'topic.{}.test.doc.pkl']
    pairs = [(_files[0], _files[1], 'train.pos'), (_files[0], _files[2], 'train.neg'), (_files[3], _files[4], 'test')]
    with open(tf_idf, 'rb') as f:
        idf_dic = pickle.load(f)
    for topic in topics:
        for pair in pairs:
            total_locals = []
            _f_path = os.path.join(out_path, pair[0].format(topic))
            print(f'loading ' + pair[0].format(topic))
            with open(_f_path, 'rb') as f:
                queries = pickle.load(f)
            _f_path = os.path.join(out_path, pair[1].format(topic))
            print(f'loading ' + pair[1].format(topic))
            with open(_f_path, 'rb') as f:
                documents = pickle.load(f)
            pbar = tqdm(total=len(documents), desc='loading ' + topic)
            for i in range(len(documents)):
                total_locals.append([])
                query = queries[i]
                doc = documents[i]
                for _x in range(len(doc)):
                    for _y in range(len(query)):
                        if _x < 200 and _y < 20 and doc[_x] == query[_y]:
                            score = idf_dic.get(query[_y], 0)
                            if score > 0:
                                total_locals[-1].append((_x, _y, round(score, 4)))
                pbar.update(1)
            pbar.close()
            with open(os.path.join(out_path, 'topic.{}.{}.idf.pkl'.format(topic, pair[2])), 'wb') as f:
                pickle.dump(total_locals, f)


if __name__ == '__main__':
    print(f"loading tokenized queries...")
    queries = load_queries(query_tokenized_path)
    print(f"loading tokenized document...")
    doc_collections = load_collections(collection_tokenized_path)
    for topic in topics:
        process_train_triples(topic)
        process_test_triples(topic)
    process_duet_local_features()
