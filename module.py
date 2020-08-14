import pandas as pd
import numpy as np
from itertools import combinations

def preprocess(df):
    menu_to_id = {menu: idx for idx, menu in enumerate(df['menu_cate'].unique())}
    id_to_menu = dict((v, k) for k, v in menu_to_id.items())

    grouped = df.groupby('sid_id')  # 매장으로 groupby 하기
    sid_menu_set = grouped['menu_cate'].apply(lambda x: x.tolist()).to_list()  # 메뉴 리스트로 만들기

    combination_menu = [list(combinations(i, 2)) for i in sid_menu_set]
    pair_menu = [j for i in combination_menu for j in i]  # 메뉴 2개씩 그룹화
    pair_menu = [(menu_to_id[menu1], menu_to_id[menu2]) for menu1, menu2 in pair_menu]

    return menu_to_id, id_to_menu, pair_menu


def create_co_matrix2(pair_menu, vocab_size):
    '''
    같은 매장에서 쓰는 같은 메뉴일경우 같은 발생으로 봄
    '''
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for i, j in pair_menu:
        co_matrix[i][j] += 1
        co_matrix[j][i] += 1

    return co_matrix


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M


def cos_similarity(x, y, eps=1e-8):
    '''코사인 유사도 산출

    :param x: 벡터
    :param y: 벡터
    :param eps: '0으로 나누기'를 방지하기 위한 작은 값
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''유사 단어 검색

    :param query: 쿼리(텍스트)
    :param word_to_id: 단어에서 단어 ID로 변환하는 딕셔너리
    :param id_to_word: 단어 ID에서 단어로 변환하는 딕셔너리
    :param word_matrix: 단어 벡터를 정리한 행렬. 각 행에 해당 단어 벡터가 저장되어 있다고 가정한다.
    :param top: 상위 몇 개까지 출력할 지 지정
    '''
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return