# text data augmentation : NLPAUG library
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas

texts = [
    "Those who can imagine anything, can create the impossible.",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent.",
]

# insert
aug = naw.ContextualWordEmbsAug(model_path = "bert-base-uncased", action = 'insert') # bert 모델을 활용해 단어 삽입
# aug = naw.ContextualWordEmbsAug(model_path = "disilbert-base-uncased", action = 'insert')
augmented_texts = aug.augment(texts) # augment 메서드를 통해 augmentation 수행

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# substitute
aug = naw.ContextualWordEmbsAug(model_path = "bert-base-uncased", action = 'substitute') 
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

#%%
# delete
aug = nac.RandomCharAug(action = 'delete') # 랜덤하게 문자 삭제
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# swap
aug = nac.RandomCharAug(action = 'swap') # 랜덤하게 문자 위치 교체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# insert
aug = nac.RandomCharAug(action = 'insert') # 랜덤하게 문자 삽입
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# substitute
aug = nac.RandomCharAug(action = 'substitute') # 랜덤하게 임의의 문자나 동의어로 대체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

#%%
# swap
aug = naw.RandomWordAug(action = 'swap') # 랜덤하게 단어 위치 교체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# delete
aug = naw.RandomWordAug(action = 'delete') # 랜덤하게 단어 삭제
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")


# insert
aug = naw.RandomWordAug(action = 'insert') # 랜덤하게 단어 삽입
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# substitute
aug = naw.RandomWordAug(action = 'substitute') # 랜덤하게 임의의 단어나 동의어로 대체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

# crop
aug = naw.RandomWordAug(action = 'crop') # 랜덤하게 자르기
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")    

#%%
# DB를 이용하여 유의어나 동의어로 대체
aug = naw.SynonymAug(aug_src = 'wordnet') # wordnet DB를 활용해 단어 대체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

aug = naw.SynonymAug(aug_src = 'ppdb') # Paraphrase DB(ppdb)를 활용해 단어 대체
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

#%%
reserved_tokens = [
    ["can", "can't", "cannot", "could"]
]

reserved_aug = naw.ReservedAug(reserved_tokens = reserved_tokens) # 입력 데이터에 포함된 단어를 특정한 단어로 대체하는 기능
augmented_texts = reserved_aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

#%%
"""Back-translation"""
# 입력 텍스트를 특정 언어로 번역한 다음 다시 본래의 언어로 번역하는 방법
# paraphrasing 효과

back_translation = naw.BackTranslationAug(
    from_model_name = 'facebook/wmt19-en-de', # 영어 -> 독일어
    to_model_name = 'facebook/wmt19-de-en'    # 독일어 -> 영어
)

augmented_texts = back_translation.augment(texts)


for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")

#%%
"""문장 요약 증강"""

summary_aug = nas.AbstSummAug(
    model_path = 't5-base'
)

augmented_texts = summary_aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("-------------------------------------")