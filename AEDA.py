import random
import pandas as pd

#AUG = []

def df_augmented(df, PUNC_RATIO=0.3,
					PUNCTUATIONS= ['.', ',', '!', '?', ';', ':']):
  AUG = []
  for i in range(len(df)):
    sent = str(df.sentence.values[i]).split()
    AUG.append(sent)

    q = random.randint(1, int(PUNC_RATIO * len(sent) + 1)) # 1부터 문장길이*0.3 까지 난수 생성
    qs = random.sample(range(0, len(sent)), q)             # 위에서 생성한 난수만큼 샘플링(문장부호 삽입위치 결정)
			
    for j in range(len(qs)):
      AUG[i].insert(qs[j], PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
    AUG[i] = " ".join(AUG[i]) # 다시 문자열로 변환
  
  AUG = pd.DataFrame(AUG, columns = ['sentence'])
  AUG['label'] = list(df['label'])
  return AUG